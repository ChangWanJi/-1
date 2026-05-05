# Sequential Active Learning for Medium Optimization (재현)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ChangWanJi/-1/blob/main/sequential_medium_optimization.ipynb)

## 📄 재현 대상 논문

> Hashizume, T., Baba, K., Matsuo, N., & Ying, B.-W. (2026).
> **Sequential active learning for medium optimization in mAb production.**
> *Journal of Bioscience and Bioengineering*, **141**(3), 210–220.
> https://doi.org/10.1016/j.jbiosc.2025.12.002
>
> 원본 GitHub: https://github.com/hashizume711/sequential-medium-optimization

CHO 세포의 IgG 단일클론항체(mAb) 생산성을 최대화하기 위해, **DOE (Design of Experiments)** 와 **머신러닝 (GBDT, MLR)** 을 반복적으로 결합하는 sequential active learning 전략을 제안한 논문입니다.

## 🎯 본 재현 프로젝트의 범위

| 단계 | 논문 내용 | 재현 여부 |
|---|---|---|
| Round 1 | 넓은 DOE → osmolality 문제 발견 | ✅ |
| Round 2 | NaCl/NaHCO₃ 분리, osmolality 통제 | ✅ |
| Round 3 | GBDT + MLR 도입, top candidate 검증 | ✅ |
| Round 4 | R2+R3 통합 학습 (n=48) | ✅ |
| Round 5 | 6개 아미노산 fine-tuning | ✅ |
| Round 6 | 통합 모델로 최적 medium 예측 | ✅ |
| Nested CV (Table S4) | 5-outer × 5-inner GridSearch | ✅ |
| SHAP / Feature importance (Fig. 7) | 해석 분석 | ✅ |

## 🚀 실행 방법

### 옵션 A — Google Colab (가장 쉬움, 추천)
1. 위쪽의 **Open in Colab** 배지 클릭
2. 메뉴에서 **런타임 → 모두 실행** (Runtime → Run all)
3. 약 2–3분 후 모든 결과/그래프 표시
4. 별도 데이터 파일 업로드 불필요

### 옵션 B — 로컬 실행
```bash
git clone https://github.com/ChangWanJi/-1.git
cd -1
pip install -r requirements.txt
jupyter notebook sequential_medium_optimization.ipynb
```

## 📊 데이터에 관한 안내

원논문의 raw 실험 측정값(IgG titer 등)은 supplementary table 형태로만 제공되고 원본 GitHub에도 포함되지 않습니다 (코드만 공개). 따라서 본 재현은 논문에서 **명시적으로 보고된 생물학적 관찰**을 함수로 인코딩한 **ground-truth simulator** 를 사용해 단일 노트북으로 self-contained 하게 만들었습니다. ML 파이프라인 (DOE → GBDT(GridSearchCV) → MLR → 검증 → 재학습 → SHAP) 자체는 **실제 데이터 CSV로 교체하면 그대로 동작**합니다.

## 📈 재현 결과

### 라운드별 IgG 생산성 (paper Fig. 6A)

| Round | 평균 IgG fold | 최고 IgG fold | 핵심 변화 |
|---|---|---|---|
| R1 | 0.42 | 1.23 | DOE만; osmolality 문제로 61% 실패 |
| R2 | 1.17 | 1.47 | NaCl/NaHCO₃ 분리; 안정화 |
| R3 | 1.27 | 1.51 | GBDT 모델 도입 |
| R4 | 1.35 | 1.54 | n=48로 재학습, GBDT가 best 달성 |
| R5 | 1.70 | 2.01 | 아미노산 fine-tuning |
| R6 | 1.69 | 1.97 | 통합 모델 (n=108) |

### 모델 성능 (논문 vs 본 재현)

| 지표 | 논문 | 본 재현 |
|---|---|---|
| Round 6 GBDT R² (test) | 0.62 | ~0.94 |
| Nested CV R² | 0.66 | ~0.95 |
| 최종 IgG fold-change | 1.7× | ~2.0× |

## 🛠️ 사용 기술 스택

| 컴포넌트 | 라이브러리 |
|---|---|
| Gradient Boosting | `sklearn.ensemble.GradientBoostingRegressor` |
| Multiple Linear Regression (RSM) | `sklearn.linear_model.LinearRegression` + `PolynomialFeatures(degree=2)` |
| Hyperparameter tuning | `sklearn.model_selection.GridSearchCV` (5-fold) |
| Nested cross-validation | `KFold` 5-outer × 5-inner |
| 모델 해석 | `shap.TreeExplainer` |

## 📝 라이선스

본 재현 코드는 학습/교육 목적으로 작성되었습니다. 원논문 © 2025 The Society for Biotechnology, Japan.
