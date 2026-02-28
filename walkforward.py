#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
walkforward.py — إطار التحقق Walk-Forward

يقسّم البيانات التاريخية إلى نوافذ متداخلة:
  [--- تدريب ---][-- اختبار --]
                 [--- تدريب ---][-- اختبار --]
                                [--- تدريب ---][-- اختبار --]

لكل نافذة:
  1. نُشغّل نفس منطق الإشارة على بيانات التدريب (داخل العينة)
  2. نختبره على بيانات الاختبار (خارج العينة — البيانات غير المرئية)
  3. نجمع مقاييس الأداء لكل نافذة
  4. نرفض مجموعة المعاملات إذا كان الأداء غير متسق عبر النوافذ

يُستخدم كأداة تشخيص — ليس كحلقة تداول فعلية.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Any

import numpy as np
import pandas as pd

from config import config

logger = logging.getLogger(__name__)


@dataclass
class WindowResult:
    """نتائج نافذة واحدة من Walk-Forward."""
    window_id: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    n_signals: int
    win_rate: float
    profit_factor: float
    expectancy_r: float        # التوقع الرياضي بوحدات R
    max_drawdown: float        # أقصى سحب (نسبة مئوية)
    is_stable: bool            # هل تجاوز الحد الأدنى المقبول؟
    notes: str = ""


@dataclass
class WalkForwardReport:
    """تقرير Walk-Forward الكامل عبر جميع النوافذ."""
    windows: List[WindowResult] = field(default_factory=list)
    n_stable: int    = 0
    n_unstable: int  = 0
    avg_win_rate: float    = 0.0
    avg_profit_factor: float = 0.0
    avg_expectancy: float  = 0.0
    win_rate_variance: float = 0.0   # تباين معدل الفوز عبر النوافذ
    is_robust: bool = False          # هل المنظومة متسقة بما يكفي؟
    rejection_reason: str = ""


class WalkForwardValidator:
    """
    يأخذ DataFrame وdستراتيجية (كـ callable) ويُنتج WalkForwardReport.

    استخدام:
        validator = WalkForwardValidator(df, signal_fn)
        report = validator.run()
        if not report.is_robust:
            # رفض المعاملات الحالية
    """

    def __init__(
        self,
        df: pd.DataFrame,
        signal_fn: Callable[[pd.DataFrame], List[Dict[str, Any]]],
        train_bars: int = config.WFV_TRAIN_BARS,
        test_bars:  int = config.WFV_TEST_BARS,
    ) -> None:
        self.df       = df.copy().reset_index(drop=True)
        self.signal_fn = signal_fn
        self.train_bars = train_bars
        self.test_bars  = test_bars
        self._min_length = train_bars + test_bars

    def run(self) -> WalkForwardReport:
        report = WalkForwardReport()

        if len(self.df) < self._min_length:
            report.rejection_reason = (
                f"بيانات غير كافية: {len(self.df)} شمعة < "
                f"{self._min_length} مطلوبة"
            )
            logger.warning("WalkForward: %s", report.rejection_reason)
            return report

        windows = self._build_windows()
        if len(windows) < config.WFV_MIN_WINDOWS:
            report.rejection_reason = (
                f"نوافذ غير كافية: {len(windows)} < {config.WFV_MIN_WINDOWS}"
            )
            return report

        for wid, (t_start, t_end, x_start, x_end) in enumerate(windows):
            wr = self._evaluate_window(wid, t_start, t_end, x_start, x_end)
            report.windows.append(wr)

        return self._aggregate(report)

    # ── بناء النوافذ ───────────────────────────────────────────────────────────

    def _build_windows(self) -> List[tuple]:
        """
        يُنشئ نوافذ بخطوة تساوي نصف test_bars (تداخل 50%).
        """
        step = self.test_bars // 2
        windows = []
        start = 0
        while start + self._min_length <= len(self.df):
            t_start = start
            t_end   = start + self.train_bars
            x_start = t_end
            x_end   = x_start + self.test_bars
            if x_end > len(self.df):
                break
            windows.append((t_start, t_end, x_start, x_end))
            start += step
        return windows

    # ── تقييم نافذة واحدة ──────────────────────────────────────────────────────

    def _evaluate_window(
        self,
        wid: int,
        t_start: int, t_end: int,
        x_start: int, x_end: int,
    ) -> WindowResult:
        test_df = self.df.iloc[x_start:x_end].copy()

        try:
            raw_signals = self.signal_fn(test_df)
        except Exception as exc:
            logger.warning("WalkForward window %d: خطأ في signal_fn: %s", wid, exc)
            raw_signals = []

        if not raw_signals:
            return WindowResult(
                window_id=wid,
                train_start=t_start, train_end=t_end,
                test_start=x_start,  test_end=x_end,
                n_signals=0, win_rate=0.0, profit_factor=0.0,
                expectancy_r=0.0, max_drawdown=0.0,
                is_stable=False, notes="لا إشارات في نافذة الاختبار",
            )

        # محاكاة بسيطة: كل إشارة تُقيَّم بعد N شموع
        metrics = self._simulate_trades(test_df, raw_signals)
        is_stable = (
            metrics["win_rate"]      >= config.WFV_MIN_WIN_RATE and
            metrics["profit_factor"] >= 1.0 and
            metrics["expectancy_r"]  >= config.EXPECTANCY_MIN
        )

        return WindowResult(
            window_id=wid,
            train_start=t_start, train_end=t_end,
            test_start=x_start,  test_end=x_end,
            n_signals=metrics["n"],
            win_rate=metrics["win_rate"],
            profit_factor=metrics["profit_factor"],
            expectancy_r=metrics["expectancy_r"],
            max_drawdown=metrics["max_drawdown"],
            is_stable=is_stable,
            notes="مستقرة" if is_stable else "غير مستقرة",
        )

    # ── محاكاة الصفقات ──────────────────────────────────────────────────────────

    @staticmethod
    def _simulate_trades(df: pd.DataFrame, signals: List[Dict]) -> Dict[str, float]:
        """
        محاكاة صفقة: إدخال عند close الإشارة → قياس نتيجة بعد 5 شموع.
        يُطبق انزلاق واقعي بسيط لكل صفقة.
        """
        rr = config.DEFAULT_RR_RATIO
        wins = losses = 0
        equity = [1.0]
        win_r_total = loss_r_total = 0.0

        for sig in signals:
            idx = sig.get("bar_index", 0)
            if idx + 5 >= len(df):
                continue

            entry_price  = float(df["close"].iloc[idx])
            future_price = float(df["close"].iloc[idx + 5])
            direction    = sig.get("direction", "BUY")

            # انزلاق عشوائي بسيط
            slippage = np.random.uniform(
                config.SLIPPAGE_MIN_PIPS, config.SLIPPAGE_MAX_PIPS
            ) * 0.0001

            if direction == "BUY":
                entry_adj  = entry_price + slippage
                pnl_pct    = (future_price - entry_adj) / entry_adj
            else:
                entry_adj  = entry_price - slippage
                pnl_pct    = (entry_adj - future_price) / entry_adj

            # تصنيف فوز/خسارة بالمقارنة مع نسبة مئوية صغيرة
            threshold = 0.0005   # 5 نقاط أساس كحد أدنى للنصر
            if pnl_pct > threshold:
                wins += 1
                win_r_total += rr
                equity.append(equity[-1] * (1 + rr * 0.01))
            else:
                losses += 1
                loss_r_total += 1.0
                equity.append(equity[-1] * (1 - 0.01))

        n = wins + losses
        win_rate      = wins / n if n > 0 else 0.0
        profit_factor = win_r_total / loss_r_total if loss_r_total > 0 else float("inf")
        expectancy_r  = (win_rate * rr) - ((1 - win_rate) * 1.0)

        # حساب أقصى سحب
        eq = np.array(equity)
        peak = np.maximum.accumulate(eq)
        dd   = (peak - eq) / peak
        max_dd = float(dd.max()) if len(dd) > 0 else 0.0

        return {
            "n": n, "win_rate": win_rate,
            "profit_factor": profit_factor,
            "expectancy_r": expectancy_r,
            "max_drawdown": max_dd,
        }

    # ── تجميع النتائج ──────────────────────────────────────────────────────────

    @staticmethod
    def _aggregate(report: WalkForwardReport) -> WalkForwardReport:
        if not report.windows:
            return report

        stable_windows = [w for w in report.windows if w.is_stable]
        report.n_stable   = len(stable_windows)
        report.n_unstable = len(report.windows) - report.n_stable

        wrs  = [w.win_rate      for w in report.windows]
        pfs  = [w.profit_factor for w in report.windows]
        exps = [w.expectancy_r  for w in report.windows]

        report.avg_win_rate      = float(np.mean(wrs))
        report.avg_profit_factor = float(np.mean([p for p in pfs if p != float("inf")]))
        report.avg_expectancy    = float(np.mean(exps))
        report.win_rate_variance = float(np.var(wrs))   # تباين → مقياس عدم الاستقرار

        # قرار الرفض/القبول
        instability = report.win_rate_variance
        enough_stable = report.n_stable >= config.WFV_MIN_WINDOWS

        if not enough_stable:
            report.is_robust = False
            report.rejection_reason = (
                f"نوافذ مستقرة: {report.n_stable} من {len(report.windows)}"
            )
        elif instability > config.WFV_MAX_INSTABILITY:
            report.is_robust = False
            report.rejection_reason = (
                f"تباين عالٍ في معدل الفوز: {instability:.3f} > {config.WFV_MAX_INSTABILITY}"
            )
        elif report.avg_expectancy < config.EXPECTANCY_MIN:
            report.is_robust = False
            report.rejection_reason = (
                f"توقع سلبي: {report.avg_expectancy:.3f} < {config.EXPECTANCY_MIN}"
            )
        else:
            report.is_robust = True

        logger.info(
            "WalkForward: قوي=%s | نوافذ مستقرة=%d/%d | توقع=%.3f",
            report.is_robust, report.n_stable, len(report.windows), report.avg_expectancy,
        )
        return report
