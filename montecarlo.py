#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
montecarlo.py — محرك Monte Carlo لاختبار متانة الاستراتيجية

يُشغّل 1000 محاكاة بترتيب عشوائي للصفقات ويُخرج:
  • توزيع أسوأ سحب (Max Drawdown Distribution)
  • النتيجة الوسيطة والمئين 5% / 95%
  • مؤشر الاستقرار (Stability Index)
  • حكم الرفض / القبول

المبدأ: إذا كانت النتائج الجيدة تعتمد على ترتيب معين للصفقات →
النظام هش وليس قوياً حقيقياً.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from config import config

logger = logging.getLogger(__name__)


@dataclass
class MonteCarloResult:
    n_simulations: int
    n_trades: int

    # توزيع السحب الأقصى
    dd_mean: float        # متوسط السحب الأقصى
    dd_median: float      # وسيط السحب الأقصى
    dd_p5: float          # مئين 5% (الأسوأ)
    dd_p95: float         # مئين 95%

    # توزيع العائد النهائي
    equity_mean: float    # متوسط رأس المال النهائي (% من البداية)
    equity_median: float
    equity_p5: float      # مئين 5% (سيناريو سيئ)
    equity_p95: float     # مئين 95% (سيناريو جيد)

    # مؤشر الاستقرار (0–1)
    stability_index: float

    # حكم النظام
    is_robust: bool
    rejection_reason: str = ""

    def summary(self) -> str:
        status = "✅ متانة مقبولة" if self.is_robust else f"❌ رفض: {self.rejection_reason}"
        return (
            f"🎲 *Monte Carlo ({self.n_simulations} محاكاة | {self.n_trades} صفقة)*\n"
            f"━━━━━━━━━━━━━━\n"
            f"السحب الأقصى الوسيط: {self.dd_median:.1%}\n"
            f"أسوأ سحب (5%): {self.dd_p5:.1%}\n"
            f"رأس المال الوسيط: {self.equity_median:.1%}\n"
            f"مئين 5%: {self.equity_p5:.1%} | 95%: {self.equity_p95:.1%}\n"
            f"مؤشر الاستقرار: {self.stability_index:.2f}/1.00\n"
            f"الحكم: {status}\n"
        )


class MonteCarloEngine:
    """
    يأخذ قائمة نتائج الصفقات (R-multiples) ويُشغّل محاكاة شاملة.

    المدخلات:
        trade_results: قائمة من قيم R-multiple
                       (موجبة = ربح، سالبة = خسارة)
                       مثال: [1.5, -1.0, 2.0, -1.0, 1.5, ...]

    استخدام:
        mc = MonteCarloEngine(trade_results)
        result = mc.run()
    """

    def __init__(
        self,
        trade_results: List[float],
        n_simulations: int = config.MC_SIMULATIONS,
        initial_capital: float = 1.0,   # مُطبَّع (1 = 100%)
        risk_per_trade: float  = 0.01,  # 1% من رأس المال لكل صفقة
    ) -> None:
        self._trades     = np.array(trade_results, dtype=float)
        self._n_sims     = n_simulations
        self._capital    = initial_capital
        self._risk       = risk_per_trade

    def run(self) -> MonteCarloResult:
        if len(self._trades) < 10:
            return MonteCarloResult(
                n_simulations=0, n_trades=len(self._trades),
                dd_mean=0, dd_median=0, dd_p5=0, dd_p95=0,
                equity_mean=1, equity_median=1, equity_p5=1, equity_p95=1,
                stability_index=0, is_robust=False,
                rejection_reason="صفقات غير كافية للمحاكاة (< 10)",
            )

        dd_list     = np.zeros(self._n_sims)
        equity_list = np.zeros(self._n_sims)

        for i in range(self._n_sims):
            # عشوائية الترتيب — جوهر الاختبار
            shuffled = np.random.permutation(self._trades)
            equity_curve = self._simulate_curve(shuffled)
            dd_list[i]     = self._max_drawdown(equity_curve)
            equity_list[i] = equity_curve[-1]

        stability = self._stability_index(dd_list, equity_list)
        result = MonteCarloResult(
            n_simulations=self._n_sims,
            n_trades=len(self._trades),
            dd_mean=float(np.mean(dd_list)),
            dd_median=float(np.median(dd_list)),
            dd_p5=float(np.percentile(dd_list, 95)),   # 95% → أسوأ سحب
            dd_p95=float(np.percentile(dd_list, 5)),   # 5% → أفضل سحب
            equity_mean=float(np.mean(equity_list)),
            equity_median=float(np.median(equity_list)),
            equity_p5=float(np.percentile(equity_list, 5)),
            equity_p95=float(np.percentile(equity_list, 95)),
            stability_index=stability,
            is_robust=False,
        )

        result = self._judge(result)
        logger.info("MonteCarlo: %s", result.summary())
        return result

    # ── الحسابات ─────────────────────────────────────────────────────────────

    def _simulate_curve(self, trade_sequence: np.ndarray) -> np.ndarray:
        """تُشغّل منحنى الأسهم لتسلسل صفقات معيّن."""
        equity = self._capital
        curve  = [equity]
        for r in trade_sequence:
            equity = equity * (1 + self._risk * r)
            equity = max(equity, 1e-9)  # منع القيم السالبة
            curve.append(equity)
        return np.array(curve)

    @staticmethod
    def _max_drawdown(equity_curve: np.ndarray) -> float:
        """يحسب أقصى سحب من الذروة إلى القاع."""
        peak = np.maximum.accumulate(equity_curve)
        with np.errstate(divide="ignore", invalid="ignore"):
            dd = np.where(peak > 0, (peak - equity_curve) / peak, 0)
        return float(np.max(dd))

    @staticmethod
    def _stability_index(dd_arr: np.ndarray, eq_arr: np.ndarray) -> float:
        """
        مؤشر الاستقرار: يقيس اتساق النتائج عبر المحاكاة.
        يدمج:
          • نسبة المحاكاة التي تنتهي بأسهم موجبة (> 1.0)
          • معامل تباين السحب (أقل = أفضل)

        يُعيد قيمة بين 0 (هش) و1 (مستقر).
        """
        profitable_ratio = float(np.mean(eq_arr > 1.0))

        dd_cv = float(np.std(dd_arr) / np.mean(dd_arr)) if np.mean(dd_arr) > 0 else 1.0
        consistency = max(0.0, 1 - min(dd_cv, 1.0))

        return round((profitable_ratio * 0.6) + (consistency * 0.4), 3)

    @staticmethod
    def _judge(result: MonteCarloResult) -> MonteCarloResult:
        """يُطبق معايير الرفض المؤسسية."""
        if result.dd_p5 > config.MC_MAX_DRAWDOWN_LIMIT:
            result.is_robust = False
            result.rejection_reason = (
                f"أسوأ سحب {result.dd_p5:.1%} > {config.MC_MAX_DRAWDOWN_LIMIT:.1%}"
            )
        elif result.stability_index < config.MC_STABILITY_MIN:
            result.is_robust = False
            result.rejection_reason = (
                f"مؤشر الاستقرار {result.stability_index:.2f} < {config.MC_STABILITY_MIN:.2f}"
            )
        elif result.equity_p5 < 0.85:   # 15% خسارة في السيناريو السيئ
            result.is_robust = False
            result.rejection_reason = (
                f"أسوأ سيناريو يُفقد {(1 - result.equity_p5):.1%} من رأس المال"
            )
        else:
            result.is_robust = True
        return result
