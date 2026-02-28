#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
slippage_model.py — نموذج الانزلاق والسبريد الواقعي

يُحاكي التكاليف الحقيقية للتنفيذ:
  • انزلاق عشوائي بتوزيع غير متماثل (الانزلاق السلبي أكثر شيوعاً)
  • توسيع السبريد في التقلب العالي
  • تأخير التنفيذ (يُؤثر على سعر الدخول)
  • محاكاة التنفيذ الجزئي عند ظروف السيولة المنخفضة

كل إشارة تمر عبر هذا النموذج قبل تسجيل نتيجتها.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from config import config

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """تفاصيل التنفيذ الواقعي لصفقة واحدة."""
    theoretical_entry: float     # سعر الدخول النظري
    actual_entry: float          # سعر الدخول الفعلي بعد الانزلاق
    spread_cost: float           # تكلفة السبريد (بوحدات السعر)
    slippage_cost: float         # تكلفة الانزلاق
    total_cost: float            # إجمالي تكلفة التنفيذ
    execution_delay_ms: int      # تأخير التنفيذ (ملي ثانية)
    fill_ratio: float            # نسبة التنفيذ (1.0 = كامل)
    is_partial: bool             # هل التنفيذ جزئي؟
    adjusted_pnl_r: float        # العائد المعدّل بعد خصم التكاليف


class SlippageModel:
    """
    نموذج تكاليف التنفيذ الواقعي.
    يُستخدم في Backtesting وWalk-Forward لتجنب التفاؤل المفرط.
    """

    def __init__(self, atr: float = 0.001, volatility_regime: str = "normal") -> None:
        """
        atr: متوسط النطاق الحقيقي للزوج الحالي
        volatility_regime: "low" | "normal" | "high"
        """
        self._atr     = atr
        self._regime  = volatility_regime
        self._rng     = np.random.default_rng()

    def simulate_entry(
        self,
        theoretical_price: float,
        direction: str,            # "BUY" | "SELL"
        is_news_candle: bool = False,
    ) -> ExecutionResult:
        """
        يُحاكي جميع تكاليف دخول صفقة واحدة.
        """
        spread   = self._compute_spread(is_news_candle)
        slip     = self._compute_slippage(direction, is_news_candle)
        delay_ms = self._compute_delay(is_news_candle)
        fill     = self._compute_fill_ratio(is_news_candle)

        total_cost = spread + abs(slip)

        # سعر الدخول الفعلي
        if direction == "BUY":
            actual_entry = theoretical_price + spread / 2 + slip
        else:
            actual_entry = theoretical_price - spread / 2 + slip

        return ExecutionResult(
            theoretical_entry=theoretical_price,
            actual_entry=actual_entry,
            spread_cost=spread,
            slippage_cost=slip,
            total_cost=total_cost,
            execution_delay_ms=delay_ms,
            fill_ratio=fill,
            is_partial=fill < 1.0,
            adjusted_pnl_r=0.0,   # يُحسب لاحقاً من SignalPipeline
        )

    def adjust_pnl(
        self,
        raw_pnl_r: float,
        exec_result: ExecutionResult,
        entry_price: float,
    ) -> float:
        """
        يُعيد R-multiple معدَّلاً بعد خصم تكاليف التنفيذ.
        """
        if entry_price == 0:
            return raw_pnl_r
        # التكلفة كنسبة من سعر الدخول → تُحوَّل إلى وحدات R
        cost_fraction = exec_result.total_cost / entry_price
        # افتراض 1% مخاطرة → 100 نقطة أساس = 1R
        cost_in_r = cost_fraction / 0.01   # ÷ نسبة المخاطرة
        return raw_pnl_r - cost_in_r

    # ── الحسابات الداخلية ─────────────────────────────────────────────────────

    def _compute_spread(self, is_news: bool) -> float:
        """السبريد يتسع في التقلب العالي والأخبار."""
        base = config.SPREAD_BASE_PIPS * 0.0001   # تحويل نقطة → سعر

        if is_news or self._regime == "high":
            multiplier = config.SPREAD_VOL_MULT
        elif self._regime == "low":
            multiplier = 0.8
        else:
            multiplier = 1.0 + self._rng.uniform(0, 0.3)   # تباين طبيعي

        return base * multiplier

    def _compute_slippage(self, direction: str, is_news: bool) -> float:
        """
        الانزلاق غير متماثل: يميل للضرر (توزيع منحرف يمين للانزلاق السلبي).
        اتجاه الانزلاق دائماً ضد المتداول (يزيد التكلفة).
        """
        if is_news or self._regime == "high":
            max_slip = config.SLIPPAGE_MAX_PIPS
        else:
            max_slip = config.SLIPPAGE_MAX_PIPS * 0.5

        # توزيع لوغاريثمي طبيعي (الانزلاق الكبير نادر لكن ممكن)
        slip_pips = self._rng.lognormal(
            mean=math.log(config.SLIPPAGE_MIN_PIPS + 0.01),
            sigma=0.5,
        )
        slip_pips = min(slip_pips, max_pip := max_slip)
        slip = slip_pips * 0.0001

        # الانزلاق دائماً يضر المتداول
        return slip if direction == "BUY" else -slip

    def _compute_delay(self, is_news: bool) -> int:
        """تأخير التنفيذ بالملي ثانية."""
        low  = config.EXEC_DELAY_MS_MIN
        high = config.EXEC_DELAY_MS_MAX * (3 if is_news else 1)
        return int(self._rng.integers(low, high))

    def _compute_fill_ratio(self, is_news: bool) -> float:
        """
        نسبة التنفيذ — في الأسواق السائلة غالباً 1.0،
        في ظروف السيولة المنخفضة قد تنخفض إلى 0.5–0.8.
        """
        if is_news:
            return float(self._rng.uniform(0.5, 0.9))
        if self._regime == "low":
            return float(self._rng.uniform(0.7, 1.0))
        return 1.0   # اكتمال التنفيذ في الظروف العادية

    # ── اكتشاف الشمعة الإخبارية ──────────────────────────────────────────────

    @staticmethod
    def is_news_candle(df_row, atr: float) -> bool:
        """
        شمعة إخبارية: نطاقها يتجاوز 2× ATR مع حجم مرتفع.
        تُستخدم لتوسيع الانزلاق تلقائياً.
        """
        candle_range = float(df_row["high"] - df_row["low"])
        volume_high  = float(df_row.get("volume", 0)) > 0   # placeholder إذا لم يتوفر حجم
        return candle_range > atr * 2.0 and volume_high
