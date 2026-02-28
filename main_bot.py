#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_bot.py — المنسق الرئيسي للنظام المؤسسي (النسخة v3)

خط الأنابيب الكامل:
  DataProvider
    → RegimeEngine          (كشف النظام السوقي)
    → RiskManager.can_trade (بوابة المخاطر الشاملة)
    → StructureEngine
    → LiquidityEngine
    → MomentumEngine
    → ConfluenceEngine      (تسجيل ثقة معدَّلة بالنظام السوقي)
    → ExpectancyEngine      (التحقق من التوقع)
    → SlippageModel         (محاكاة التنفيذ)
    → PerformanceTracker    (تسجيل + تقييم)
    → ReportEngine          (تقرير مؤسسي دوري)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
from ta.trend import ADXIndicator

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

from config import config
from data_layer import DataProvider
from structure_engine import StructureEngine, higher_tf_bias
from liquidity_engine import LiquidityEngine
from momentum_engine import MomentumEngine
from confluence_engine import ConfluenceEngine, SignalResult
from regime_engine import RegimeEngine, RegimeResult, Regime
from risk_manager import RiskManager
from expectancy_engine import ExpectancyEngine
from slippage_model import SlippageModel
from performance_tracker import PerformanceTracker, TradeRecord, make_trade_id
from report_engine import ReportEngine
from montecarlo import MonteCarloEngine
from walkforward import WalkForwardValidator

logger = logging.getLogger(__name__)


# ── خط أنابيب الإشارة ─────────────────────────────────────────────────────────

class SignalPipeline:
    """
    الكائن المركزي الذي يُشغّل التحليل الكامل لرمز واحد.
    يدمج جميع المحركات في تسلسل منطقي وقابل للتدقيق.
    """

    def __init__(self, data_provider: DataProvider, risk: RiskManager) -> None:
        self._dp   = data_provider
        self._risk = risk

    async def run(self, symbol_key: str) -> Optional[Dict[str, Any]]:

        # ── 1. جلب البيانات والتحقق ──────────────────────────────────────────
        df_htf = await self._dp.get_validated(symbol_key, config.HIGHER_TF, config.HIGHER_PERIOD)
        df_ltf = await self._dp.get_validated(symbol_key, config.LOWER_TF,  config.LOWER_PERIOD)
        if df_htf is None or df_ltf is None:
            return None

        # ── 2. كشف النظام السوقي ────────────────────────────────────────────
        try:
            regime_result: RegimeResult = RegimeEngine(df_htf).detect()
            self._risk.set_regime(regime_result)
        except ValueError as e:
            logger.debug("%s: regime skip — %s", symbol_key, e)
            return None

        # ── 3. حساب ADX و ATR ───────────────────────────────────────────────
        adx_htf = self._compute_adx(df_htf)
        atr_ltf = self._compute_atr(df_ltf)

        # ── 4. بوابة المخاطر الشاملة ─────────────────────────────────────────
        if not await self._risk.can_trade(symbol_key, adx_htf, atr_ltf, regime_result):
            return None

        # ── 5. تحليل الهيكل والسيولة والزخم ──────────────────────────────────
        try:
            structure_res = StructureEngine(df_ltf).analyse()
            liquidity_res = LiquidityEngine(df_ltf).analyse()
            momentum_res  = MomentumEngine(df_ltf).analyse()
        except Exception as exc:
            logger.warning("%s: خطأ في المحركات — %s", symbol_key, exc)
            return None

        # رفض إذا كان السوق في تعزيز
        if structure_res.is_consolidating:
            return None

        # ── 6. تحيز الإطار الأعلى ────────────────────────────────────────────
        htf_trend = higher_tf_bias(df_htf)

        # ── 7. التسجيل في Confluence مع مراعاة النظام السوقي ────────────────
        conf_engine = ConfluenceEngine(
            structure=structure_res,
            liquidity=liquidity_res,
            momentum=momentum_res,
            htf_trend=htf_trend,
            adx_htf=adx_htf,
        )
        signal: SignalResult = conf_engine.score()

        # ── 8. عتبة ثقة ديناميكية (تتكيف مع النظام + الأداء + السحب) ────────
        threshold = self._risk.adaptive_confidence_threshold(regime_result)
        if signal.direction == "NEUTRAL" or signal.confidence < threshold:
            return None

        # ── 9. محاكاة الانزلاق ───────────────────────────────────────────────
        regime_str = "high" if regime_result.atr_ratio > config.REGIME_ATR_HV_MULT else "normal"
        slip_model = SlippageModel(atr=atr_ltf, volatility_regime=regime_str)
        last_row   = df_ltf.iloc[-1]
        is_news    = SlippageModel.is_news_candle(last_row, atr_ltf)
        exec_res   = slip_model.simulate_entry(
            theoretical_price=float(df_ltf["close"].iloc[-1]),
            direction=signal.direction,
            is_news_candle=is_news,
        )

        # ── 10. بناء حمولة الإشارة الكاملة ──────────────────────────────────
        return {
            "symbol":        symbol_key,
            "direction":     signal.direction,
            "confidence":    signal.confidence,
            "threshold_used": threshold,
            "expiry":        signal.expiry,
            "price":         float(df_ltf["close"].iloc[-1]),
            "actual_entry":  exec_res.actual_entry,
            "slippage_cost": exec_res.slippage_cost,
            "spread_cost":   exec_res.spread_cost,
            "is_news_candle": is_news,
            "timestamp":     datetime.now(timezone.utc).isoformat(),
            "htf_trend":     htf_trend,
            "adx":           adx_htf,
            "atr":           atr_ltf,
            "regime":        regime_result.regime,
            "regime_persist": regime_result.directional_persistence,
            "session":       regime_result.session,
            "bos_bullish":   structure_res.bos_bullish,
            "bos_bearish":   structure_res.bos_bearish,
            "choch":         structure_res.choch,
            "sweep_bull":    liquidity_res.sweep_bullish,
            "sweep_bear":    liquidity_res.sweep_bearish,
            "momentum_bias": momentum_res.momentum_bias,
            "rsi":           momentum_res.rsi,
            "breakdown":     signal.breakdown,
        }

    # ── حسابات مساعدة ────────────────────────────────────────────────────────

    @staticmethod
    def _compute_adx(df: pd.DataFrame, window: int = 14) -> float:
        try:
            s = ADXIndicator(high=df["high"], low=df["low"], close=df["close"], window=window).adx().dropna()
            return float(s.iloc[-1]) if len(s) else 0.0
        except Exception:
            return 0.0

    @staticmethod
    def _compute_atr(df: pd.DataFrame, window: int = 14) -> float:
        try:
            tr = pd.concat([
                df["high"] - df["low"],
                (df["high"] - df["close"].shift()).abs(),
                (df["low"]  - df["close"].shift()).abs(),
            ], axis=1).max(axis=1)
            return float(tr.rolling(window).mean().iloc[-1])
        except Exception:
            return 0.001


# ── بوت تيليجرام ──────────────────────────────────────────────────────────────

class TradingBot:
    """واجهة تيليجرام مع خط أنابيب الإشارة والتقارير المؤسسية."""

    def __init__(self) -> None:
        self._data_provider = DataProvider()
        self._risk          = RiskManager()
        self._tracker       = PerformanceTracker()
        self._pipeline      = SignalPipeline(self._data_provider, self._risk)
        self._reporter      = ReportEngine(self._tracker)
        self._subscribers   = self._load_subscribers()
        self._app: Optional[Application] = None
        # نتائج R-multiple لـ Monte Carlo
        self._trade_r_log: List[float] = []

    # ── دورة حياة البوت ───────────────────────────────────────────────────────

    def run(self) -> None:
        self._app = Application.builder().token(config.TOKEN).build()
        self._app.add_handler(CommandHandler("start",  self._cmd_start))
        self._app.add_handler(CommandHandler("stats",  self._cmd_stats))
        self._app.add_handler(CommandHandler("report", self._cmd_report))
        self._app.add_handler(CommandHandler("mc",     self._cmd_monte_carlo))
        self._app.add_handler(CommandHandler("risk",   self._cmd_risk))
        self._app.add_handler(CallbackQueryHandler(self._btn_handler))

        # فحص دوري كل 5 دقائق
        self._app.job_queue.run_repeating(self._scan_all, interval=300, first=15)
        # تقرير مؤسسي كل 6 ساعات
        self._app.job_queue.run_repeating(self._periodic_report, interval=21600, first=300)

        logger.info("البوت قيد التشغيل...")
        self._app.run_polling()

    # ── أوامر المستخدم ────────────────────────────────────────────────────────

    async def _cmd_start(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        keyboard = [
            [InlineKeyboardButton("🔔 اشتراك",  callback_data="sub"),
             InlineKeyboardButton("🔕 إلغاء",   callback_data="unsub")],
            [InlineKeyboardButton("📊 إحصائيات", callback_data="stats"),
             InlineKeyboardButton("🏦 تقرير مؤسسي", callback_data="report")],
            [InlineKeyboardButton("🎲 Monte Carlo", callback_data="mc"),
             InlineKeyboardButton("🛡️ المخاطر", callback_data="risk")],
        ]
        await update.message.reply_text(
            "🤖 *Legendary Signal Engine v3 — المستوى المؤسسي*\n"
            "━━━━━━━━━━━━━━\n"
            "✅ كشف النظام السوقي\n"
            "✅ Walk-Forward Validation\n"
            "✅ محرك التوقع وخطر الإفلاس\n"
            "✅ Monte Carlo للمتانة\n"
            "✅ محاكاة انزلاق واقعية\n"
            "✅ Sharpe / Sortino / Calmar\n"
            "✅ حماية رأس المال متدرجة\n",
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="Markdown",
        )

    async def _cmd_stats(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        metrics = self._tracker.get_metrics()
        exp     = self._risk.get_expectancy_snapshot()
        text = (
            f"📊 *الإحصائيات*\n"
            f"الصفقات: {metrics.total_trades} | معدل الفوز: {metrics.win_rate}%\n"
            f"عامل الربح: {metrics.profit_factor}\n"
            f"التوقع: {exp.expectancy_r:+.4f} R\n"
            f"خطر الإفلاس: {exp.risk_of_ruin:.2%}\n"
            f"كسر كيلي الآمن: {exp.safe_kelly:.1%}\n"
        )
        await update.message.reply_text(text, parse_mode="Markdown")

    async def _cmd_report(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text("⏳ جاري إنشاء التقرير المؤسسي...")
        text = self._reporter.format_telegram()
        await update.message.reply_text(text, parse_mode="Markdown")

    async def _cmd_monte_carlo(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if len(self._trade_r_log) < 10:
            await update.message.reply_text("⚠️ بيانات غير كافية لـ Monte Carlo (نحتاج 10+ صفقات).")
            return
        await update.message.reply_text("⏳ تشغيل 1000 محاكاة...")
        mc = MonteCarloEngine(self._trade_r_log)
        result = mc.run()
        await update.message.reply_text(result.summary(), parse_mode="Markdown")

    async def _cmd_risk(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        text = self._risk.status_summary()
        await update.message.reply_text(text, parse_mode="Markdown")

    # ── معالج الأزرار ─────────────────────────────────────────────────────────

    async def _btn_handler(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        await query.answer()
        uid = query.from_user.id

        if query.data == "sub":
            self._subscribers.add(uid)
            self._save_subscribers()
            await query.edit_message_text("✅ تم الاشتراك.")
        elif query.data == "unsub":
            self._subscribers.discard(uid)
            self._save_subscribers()
            await query.edit_message_text("🔕 تم إلغاء الاشتراك.")
        elif query.data == "stats":
            metrics = self._tracker.get_metrics()
            await query.edit_message_text(
                f"معدل الفوز: {metrics.win_rate}% | P.F: {metrics.profit_factor}",
                parse_mode="Markdown",
            )
        elif query.data == "report":
            await query.edit_message_text("⏳ جاري إنشاء التقرير...")
            text = self._reporter.format_telegram()
            await ctx.bot.send_message(chat_id=uid, text=text, parse_mode="Markdown")
        elif query.data == "mc":
            if len(self._trade_r_log) < 10:
                await query.edit_message_text("⚠️ بيانات غير كافية للمحاكاة.")
                return
            mc = MonteCarloEngine(self._trade_r_log)
            result = mc.run()
            await query.edit_message_text(result.summary(), parse_mode="Markdown")
        elif query.data == "risk":
            await query.edit_message_text(self._risk.status_summary(), parse_mode="Markdown")

    # ── المسح الدوري ──────────────────────────────────────────────────────────

    async def _scan_all(self, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._subscribers:
            return
        for symbol_key in config.SYMBOLS:
            try:
                result = await self._pipeline.run(symbol_key)
                if result:
                    await self._emit_signal(ctx, result)
            except Exception as exc:
                logger.error("خطأ في %s: %s", symbol_key, exc, exc_info=True)

    async def _periodic_report(self, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """تقرير مؤسسي دوري للمشرف فقط."""
        try:
            text = self._reporter.format_telegram()
            await ctx.bot.send_message(chat_id=config.ADMIN_ID, text=text, parse_mode="Markdown")
        except Exception as e:
            logger.warning("فشل إرسال التقرير الدوري: %s", e)

    # ── إصدار الإشارة ─────────────────────────────────────────────────────────

    async def _emit_signal(self, ctx: ContextTypes.DEFAULT_TYPE, data: Dict[str, Any]) -> None:
        trade_id = make_trade_id(data["symbol"])

        record = TradeRecord(
            trade_id    = trade_id,
            symbol      = data["symbol"],
            direction   = data["direction"],
            entry_price = data["actual_entry"],   # سعر الدخول الفعلي بعد الانزلاق
            confidence  = data["confidence"],
            expiry      = data["expiry"],
            timestamp   = data["timestamp"],
            htf_trend   = data["htf_trend"],
            adx         = data["adx"],
            bos         = "bullish" if data["bos_bullish"] else "bearish" if data["bos_bearish"] else "none",
            choch       = data["choch"],
            sweep       = data["sweep_bull"] or data["sweep_bear"],
            momentum_bias = data["momentum_bias"],
            result      = "pending",
        )
        self._tracker.log_signal(record)
        await self._risk.record_signal(data["symbol"])

        emoji = "🟢" if data["direction"] == "BUY" else "🔴"
        bos_str = (
            "BOS صاعد" if data["bos_bullish"] else
            "BOS هابط" if data["bos_bearish"] else "لا يوجد"
        )
        msg = (
            f"{emoji} *إشارة {data['direction']}* {emoji}\n"
            f"━━━━━━━━━━━━━━\n"
            f"💰 الزوج: `{data['symbol']}`\n"
            f"🎯 الثقة: `{data['confidence']}%` (حد: {data['threshold_used']:.0f}%)\n"
            f"💵 سعر الدخول الفعلي: `{data['actual_entry']:.5f}`\n"
            f"💸 تكلفة الانزلاق: `{data['slippage_cost']*10000:.1f} pips`\n"
            f"⏳ انتهاء: `{data['expiry']}`\n"
            f"━━━━━━━━━━━━━━\n"
            f"🌐 النظام: `{data['regime']}`\n"
            f"📍 الجلسة: `{data['session']}`\n"
            f"📊 الاتجاه HTF: `{data['htf_trend']}`\n"
            f"📐 ADX: `{data['adx']:.1f}` | ATR: `{data['atr']:.5f}`\n"
            f"🔧 الهيكل: `{bos_str}` | CHOCH: `{data['choch']}`\n"
            f"⚡ الزخم: `{data['momentum_bias']}`\n"
            f"📰 شمعة إخبارية: {'نعم ⚠️' if data['is_news_candle'] else 'لا'}\n"
            f"━━━━━━━━━━━━━━\n"
            f"🆔 `{trade_id}`\n"
            f"_/report للتقرير المؤسسي_"
        )

        for uid in list(self._subscribers):
            try:
                await ctx.bot.send_message(chat_id=uid, text=msg, parse_mode="Markdown")
            except Exception as exc:
                logger.warning("فشل الإرسال إلى %s: %s", uid, exc)

    # ── استمرارية الاشتراكين ──────────────────────────────────────────────────

    def _load_subscribers(self):
        if os.path.exists(config.SUBSCRIBERS_FILE):
            try:
                with open(config.SUBSCRIBERS_FILE) as f:
                    return set(json.load(f))
            except Exception:
                pass
        return set()

    def _save_subscribers(self) -> None:
        with open(config.SUBSCRIBERS_FILE, "w") as f:
            json.dump(list(self._subscribers), f)


# ── نقطة الدخول ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("engine_v3.log", encoding="utf-8"),
        ],
    )
    bot = TradingBot()
    bot.run()
