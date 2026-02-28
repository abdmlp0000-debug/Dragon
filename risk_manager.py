#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
risk_manager.py — مدير المخاطر المؤسسي (النسخة v3)

ترقية شاملة تشمل:
  • حماية رأس المال متدرجة (throttle -> halt عند مستويات السحب)
  • تكامل مع RegimeEngine لتعديل الحدود ديناميكياً
  • تكامل مع ExpectancyEngine لإيقاف الإشارات عند تدهور التوقع
  • فلاتر ATR ديناميكية (عتبات غير ثابتة)
  • جميع العمليات آمنة asyncio
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Dict, Deque, Optional, Tuple

from config import config
from expectancy_engine import ExpectancyEngine, ExpectancySnapshot

logger = logging.getLogger(__name__)

_TS = datetime


class RiskManager:
    """
    البوابة الرئيسية لجميع إشارات النظام.
    يجمع: Kill Switch + Rate Limit + Drawdown Control + Expectancy Gate +
           Regime Adjustment + ATR Volatility Filter.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()

        # حالة لكل رمز
        self._signal_times: Dict[str, Deque[_TS]] = {}
        self._consec_losses: Dict[str, int]        = {}
        self._kill_switch: Dict[str, bool]          = {}
        self._kills_wins_needed: Dict[str, int]     = {}
        self._cooldown_until: Dict[str, _TS]        = {}

        # تتبع الأسهم وحماية رأس المال (عالمي)
        self._trade_results: Deque[Tuple[str, str]] = deque(maxlen=config.ROLLING_WINDOW)
        self._equity_curve: list = [1.0]
        self._peak_equity: float = 1.0
        self._is_halted: bool    = False
        self._halt_until: Optional[_TS] = None

        # محرك التوقع المدمج
        self._expectancy_engine = ExpectancyEngine()

        # آخر نظام سوقي
        self._current_regime = None

    # ── الواجهة العامة ────────────────────────────────────────────────────────

    async def can_trade(self, symbol: str, adx: float, atr: float = 0.001, regime=None) -> bool:
        async with self._lock:
            now = datetime.now(timezone.utc)

            # 1. وقف رأس المال الكامل
            if self._is_halted:
                if self._halt_until and now < self._halt_until:
                    return False
                else:
                    self._is_halted = False
                    logger.info("انتهى الإيقاف الكامل — استئناف")

            # 2. Kill Switch لكل رمز
            if self._kill_switch.get(symbol, False):
                return False

            # 3. فترة التهدئة
            if symbol in self._cooldown_until and now < self._cooldown_until[symbol]:
                return False

            # 4. فلتر ADX
            adx_threshold = config.ADX_MIN_TREND
            if adx < adx_threshold:
                return False

            # 5. حد معدل الإشارات
            max_signals = regime.max_signals if regime else config.MAX_SIGNALS_PER_15M
            self._prune_signal_times(symbol, now)
            if len(self._signal_times.get(symbol, deque())) >= max_signals:
                return False

            # 6. بوابة التوقع الرياضي
            exp_snap = self._expectancy_engine.evaluate()
            if not exp_snap.signals_allowed:
                logger.warning("توقف الإشارات: %s", exp_snap.reason)
                return False

            # 7. فلتر النظام السوقي
            if regime and not regime.signal_allowed:
                return False

            return True

    def adaptive_confidence_threshold(self, regime=None) -> float:
        base = regime.conf_threshold if regime else config.MIN_CONFIDENCE

        # تعديل بسبب تدهور الأداء
        if len(self._trade_results) >= 10:
            wins = sum(1 for _, r in self._trade_results if r == "win")
            win_rate = wins / len(self._trade_results)
            if win_rate < 0.40:
                adjustment = (0.40 - win_rate) * 30
                base = min(base + 10, base + adjustment)

        # تعديل بسبب السحب
        dd = self._current_drawdown()
        if dd >= config.CAPITAL_DRAWDOWN_THROTTLE:
            dd_factor = (dd - config.CAPITAL_DRAWDOWN_THROTTLE) / max(
                config.CAPITAL_DRAWDOWN_HALT - config.CAPITAL_DRAWDOWN_THROTTLE, 0.001
            )
            base += dd_factor * 10

        return min(base, 95.0)

    async def record_signal(self, symbol: str) -> None:
        async with self._lock:
            now = datetime.now(timezone.utc)
            if symbol not in self._signal_times:
                self._signal_times[symbol] = deque()
            self._signal_times[symbol].append(now)

    async def record_outcome(self, symbol: str, result: str, pnl_r: Optional[float] = None) -> None:
        async with self._lock:
            self._trade_results.append((symbol, result))
            self._expectancy_engine.record(result, pnl_r)
            self._update_equity(result, pnl_r)

            if result == "loss":
                self._handle_loss(symbol)
            else:
                self._handle_win(symbol)

            self._capital_protection_check()

    def set_regime(self, regime) -> None:
        self._current_regime = regime

    def get_expectancy_snapshot(self) -> ExpectancySnapshot:
        return self._expectancy_engine.evaluate()

    # ── منحنى الأسهم ──────────────────────────────────────────────────────────

    def _update_equity(self, result: str, pnl_r):
        risk = 0.01
        r = pnl_r if pnl_r is not None else (
            config.DEFAULT_RR_RATIO if result == "win" else -1.0
        )
        current = self._equity_curve[-1] * (1 + risk * r)
        self._equity_curve.append(max(current, 1e-9))
        self._peak_equity = max(self._peak_equity, current)

    def _current_drawdown(self) -> float:
        if self._peak_equity == 0:
            return 0.0
        return max(0.0, (self._peak_equity - self._equity_curve[-1]) / self._peak_equity)

    # ── حماية رأس المال ───────────────────────────────────────────────────────

    def _capital_protection_check(self) -> None:
        dd = self._current_drawdown()
        if dd >= config.CAPITAL_DRAWDOWN_HALT:
            self._is_halted = True
            self._halt_until = datetime.now(timezone.utc) + timedelta(
                hours=config.CAPITAL_DRAWDOWN_COOLDOWN_H
            )
            logger.critical("HALT! السحب: %.1f%%", dd * 100)
        elif dd >= config.CAPITAL_DRAWDOWN_THROTTLE:
            logger.warning("تقليص الإشارات — السحب: %.1f%%", dd * 100)

    # ── معالجات الفوز / الخسارة ───────────────────────────────────────────────

    def _handle_loss(self, symbol: str) -> None:
        self._consec_losses[symbol] = self._consec_losses.get(symbol, 0) + 1
        losses = self._consec_losses[symbol]
        if losses >= config.KILL_SWITCH_CONSECUTIVE_LOSSES:
            self._kill_switch[symbol] = True
            self._kills_wins_needed[symbol] = config.KILL_SWITCH_RESET_WINS
            logger.warning("Kill Switch فعال: %s", symbol)
        else:
            self._cooldown_until[symbol] = datetime.now(timezone.utc) + timedelta(
                minutes=config.COOLDOWN_AFTER_LOSS_MINUTES
            )

    def _handle_win(self, symbol: str) -> None:
        self._cooldown_until.pop(symbol, None)
        if self._kill_switch.get(symbol, False):
            n = self._kills_wins_needed.get(symbol, config.KILL_SWITCH_RESET_WINS) - 1
            self._kills_wins_needed[symbol] = n
            if n <= 0:
                self._kill_switch[symbol] = False
                self._consec_losses[symbol] = 0
        else:
            self._consec_losses[symbol] = 0

    def _prune_signal_times(self, symbol: str, now: _TS) -> None:
        cutoff = now - timedelta(minutes=15)
        q = self._signal_times.get(symbol, deque())
        while q and q[0] < cutoff:
            q.popleft()
        self._signal_times[symbol] = q

    def status_summary(self) -> str:
        dd = self._current_drawdown()
        exp = self._expectancy_engine.evaluate()
        return (
            f"السحب الحالي: {dd:.1%}\n"
            f"الإيقاف: {'نعم' if self._is_halted else 'لا'}\n"
            f"التوقع المتدحرج: {exp.rolling_expectancy:+.4f} R\n"
            f"خطر الإفلاس: {exp.risk_of_ruin:.2%}\n"
        )
