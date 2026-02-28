#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
regime_engine.py — محرك كشف النظام السوقي (Market Regime Detection)

يصنّف السوق إلى أحد خمسة أنظمة:
  • TRENDING_BULL   — اتجاه صاعد واضح (ADX مرتفع + استمرارية)
  • TRENDING_BEAR   — اتجاه هابط واضح
  • RANGING         — سوق عرضي (ADX منخفض + ضغط)
  • HIGH_VOL        — توسع تقلبات حاد (أحداث إخبارية، شموع ضخمة)
  • LOW_VOL         — ضغط تقلبات (ما قبل الاندفاع)
  • SESSION_SHIFT   — تحول ناجم عن تداخل جلسات

كل نظام يُعدّل:
  • عتبة الثقة المطلوبة
  • الحد الأقصى للإشارات
  • تسامح المخاطرة
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from config import config

logger = logging.getLogger(__name__)


# ── تعريف الأنظمة السوقية ─────────────────────────────────────────────────────

class Regime:
    TRENDING_BULL = "TRENDING_BULL"
    TRENDING_BEAR = "TRENDING_BEAR"
    RANGING       = "RANGING"
    HIGH_VOL      = "HIGH_VOL"
    LOW_VOL       = "LOW_VOL"
    SESSION_SHIFT = "SESSION_SHIFT"
    UNKNOWN       = "UNKNOWN"


@dataclass
class RegimeResult:
    regime: str              # أحد قيم Regime.*
    adx: float               # قيمة ADX الحالية
    atr: float               # متوسط النطاق الحقيقي
    atr_ratio: float         # ATR الحالي / ATR المتوسط (مؤشر توسع/ضغط)
    directional_persistence: float  # مدى استمرارية الاتجاه (0–1)
    session: str             # الجلسة الحالية: "tokyo" | "london" | "ny" | "overlap" | "off"

    # المعاملات الديناميكية المُعدَّلة حسب النظام
    conf_threshold: float    # عتبة الثقة المطلوبة
    max_signals: int         # حد الإشارات كل 15 دقيقة
    risk_multiplier: float   # مضاعف المخاطرة (1.0 = عادي، < 1 = محافظ)
    signal_allowed: bool     # هل الإشارات مسموحة في هذا النظام؟

    def summary(self) -> str:
        return (
            f"النظام: {self.regime} | ADX: {self.adx:.1f} | "
            f"ATR×: {self.atr_ratio:.2f} | جلسة: {self.session}"
        )


class RegimeEngine:
    """
    يحلل DataFrame واحد (إطار زمني أعلى يُفضَّل) ويُصدر RegimeResult.
    يجب استدعاؤه قبل ConfluenceEngine ليضبط المعاملات ديناميكياً.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        if len(df) < 50:
            raise ValueError("RegimeEngine يحتاج 50 شمعة على الأقل")
        self.df = df

    def detect(self) -> RegimeResult:
        adx   = self._compute_adx()
        atr, atr_ratio = self._compute_atr_ratio()
        persist = self._directional_persistence()
        session = self._current_session()

        regime = self._classify(adx, atr_ratio, persist)

        conf_threshold, max_signals, risk_mult, allowed = self._regime_params(regime, atr_ratio)

        result = RegimeResult(
            regime=regime,
            adx=adx,
            atr=atr,
            atr_ratio=atr_ratio,
            directional_persistence=persist,
            session=session,
            conf_threshold=conf_threshold,
            max_signals=max_signals,
            risk_multiplier=risk_mult,
            signal_allowed=allowed,
        )
        logger.info("نظام السوق: %s", result.summary())
        return result

    # ── الحسابات الداخلية ─────────────────────────────────────────────────────

    def _compute_adx(self, window: int = 14) -> float:
        """ADX بدون ta library — لتفادي تبعية دائرية."""
        high  = self.df["high"].values
        low   = self.df["low"].values
        close = self.df["close"].values
        n = len(close)
        if n < window + 2:
            return 0.0

        tr    = np.zeros(n)
        dm_p  = np.zeros(n)
        dm_m  = np.zeros(n)

        for i in range(1, n):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i - 1])
            lc = abs(low[i]  - close[i - 1])
            tr[i] = max(hl, hc, lc)

            up   = high[i]  - high[i - 1]
            down = low[i - 1] - low[i]
            dm_p[i] = up   if (up > down and up > 0)   else 0.0
            dm_m[i] = down if (down > up and down > 0) else 0.0

        def smooth(arr, w):
            s = np.zeros(len(arr))
            s[w] = arr[1: w + 1].sum()
            for i in range(w + 1, len(arr)):
                s[i] = s[i - 1] - s[i - 1] / w + arr[i]
            return s

        atr_s  = smooth(tr,   window)
        dmp_s  = smooth(dm_p, window)
        dmm_s  = smooth(dm_m, window)

        with np.errstate(divide="ignore", invalid="ignore"):
            di_p = np.where(atr_s > 0, 100 * dmp_s / atr_s, 0)
            di_m = np.where(atr_s > 0, 100 * dmm_s / atr_s, 0)
            dx   = np.where((di_p + di_m) > 0, 100 * np.abs(di_p - di_m) / (di_p + di_m), 0)

        # متوسط متحرك لـ DX
        adx_arr = np.zeros(n)
        adx_arr[2 * window] = dx[window: 2 * window + 1].mean()
        for i in range(2 * window + 1, n):
            adx_arr[i] = (adx_arr[i - 1] * (window - 1) + dx[i]) / window

        return float(adx_arr[-1])

    def _compute_atr_ratio(self, window: int = 14) -> tuple[float, float]:
        """نسبة ATR الحالي إلى متوسط ATR التاريخي (مقياس التقلب)."""
        high  = self.df["high"]
        low   = self.df["low"]
        close = self.df["close"]

        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs(),
        ], axis=1).max(axis=1)

        atr_current = float(tr.iloc[-window:].mean())
        atr_hist    = float(tr.iloc[-50:].mean()) if len(tr) >= 50 else atr_current
        ratio = atr_current / atr_hist if atr_hist > 0 else 1.0
        return atr_current, ratio

    def _directional_persistence(self, window: int = 20) -> float:
        """
        نسبة الشموع التي تتحرك في نفس اتجاه الاتجاه العام.
        0 → لا اتجاه، 1 → اتجاه نقي.
        """
        closes = self.df["close"].iloc[-window:].values
        if len(closes) < 3:
            return 0.0
        returns = np.diff(closes)
        positive = (returns > 0).sum()
        negative = (returns < 0).sum()
        total    = positive + negative
        if total == 0:
            return 0.0
        dominant = max(positive, negative)
        return float(dominant / total)

    @staticmethod
    def _current_session() -> str:
        """
        تحديد الجلسة النشطة بناءً على الوقت UTC.
        طوكيو: 00:00–09:00 | لندن: 07:00–16:00 | نيويورك: 13:00–22:00
        """
        hour = datetime.now(timezone.utc).hour
        in_tokyo  = 0  <= hour < 9
        in_london = 7  <= hour < 16
        in_ny     = 13 <= hour < 22

        active = sum([in_tokyo, in_london, in_ny])
        if active >= 2:
            return "overlap"      # تداخل جلستين → أعلى سيولة
        if in_tokyo:
            return "tokyo"
        if in_london:
            return "london"
        if in_ny:
            return "ny"
        return "off"              # خارج ساعات التداول الرئيسية

    # ── تصنيف النظام ──────────────────────────────────────────────────────────

    @staticmethod
    def _classify(adx: float, atr_ratio: float, persist: float) -> str:
        """
        منطق التصنيف الهرمي:
        1. التقلب العالي له الأولوية (حماية رأس المال)
        2. ثم التقلب المنخفض جداً (ضغط = خطر اندفاع مفاجئ)
        3. ثم التوجه بالاتجاه (ADX + استمرارية)
        4. الافتراضي: عرضي
        """
        if atr_ratio >= config.REGIME_ATR_HV_MULT:
            return Regime.HIGH_VOL

        if atr_ratio <= config.REGIME_ATR_LV_MULT:
            return Regime.LOW_VOL

        if adx >= config.REGIME_ADX_TRENDING and persist >= 0.60:
            # تحديد الاتجاه الصاعد / الهابط لاحقاً في ConfluenceEngine
            return Regime.TRENDING_BULL   # (سيُحدَّد الاتجاه من higher_tf_bias)

        if adx <= config.REGIME_ADX_RANGING:
            return Regime.RANGING

        return Regime.RANGING   # النظام الافتراضي الأكثر تحفظاً

    @staticmethod
    def _regime_params(regime: str, atr_ratio: float) -> tuple[float, int, float, bool]:
        """
        تُعيد: (conf_threshold, max_signals_per_15m, risk_multiplier, signal_allowed)
        """
        if regime == Regime.HIGH_VOL:
            # تقلب عالٍ: شروط أشد، إشارات أقل
            return config.REGIME_CONF_HIGH_VOL, config.REGIME_RATE_HIGH_VOL, 0.5, True

        if regime == Regime.LOW_VOL:
            # ضغط: نقص سيولة → حذر
            return config.REGIME_CONF_LOW_VOL, config.REGIME_RATE_LOW_VOL, 0.7, True

        if regime in (Regime.TRENDING_BULL, Regime.TRENDING_BEAR):
            # اتجاه واضح: أفضل ظروف للإشارات
            return config.REGIME_CONF_TRENDING, config.REGIME_RATE_TRENDING, 1.0, True

        if regime == Regime.RANGING:
            # عرضي: أشد صرامةً وأقل إشارات
            return config.REGIME_CONF_RANGING, config.REGIME_RATE_RANGING, 0.6, True

        if regime == Regime.SESSION_SHIFT:
            return config.REGIME_CONF_HIGH_VOL, 1, 0.5, True

        # UNKNOWN → لا إشارات
        return 90.0, 0, 0.0, False
