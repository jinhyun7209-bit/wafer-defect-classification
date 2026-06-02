# wafer-defect-classification
# WM-811K 웨이퍼 맵 불량 패턴 분류

## 프로젝트 목적
반도체 웨이퍼 맵 데이터를 분석하여 불량 패턴을 자동으로 분류하는 머신러닝 모델 구현

## 사용 기술
Python, pandas, numpy, scikit-learn, scikit-image, matplotlib

## 데이터
- 출처: WM-811K Wafer Map Dataset (Kaggle)
- 811,457개 웨이퍼 맵
- 불량 패턴 8가지: Center, Donut, Edge-Loc, Edge-Ring, Loc, Near-full, Random, Scratch
- 정상(none) 785,938개 제외, 불량 25,519개 사용
- Edge-Ring 9,680개 vs Near-full 149개 (65배 클래스 불균형)

## 과정
1. failureType 전처리 (중첩 리스트 → 텍스트 변환)
2. 웨이퍼 맵 32×32로 리사이즈 후 1차원 변환
3. class_weight='balanced'로 클래스 불균형 대응
4. RandomForest 모델 학습
5. 혼동 행렬, 오분류 분석, 해상도 실험 수행

## 실험 결과

### 모델 성능 (32×32 기준)
| 패턴 | Precision | Recall | F1 |
|------|-----------|--------|----|
| Center | 0.93 | 0.95 | 0.94 |
| Donut | 0.89 | 0.69 | 0.78 |
| Edge-Loc | 0.69 | 0.93 | 0.80 |
| Edge-Ring | 0.99 | 0.95 | 0.97 |
| Loc | 0.71 | 0.63 | 0.67 |
| Near-full | 1.00 | 0.77 | 0.87 |
| Random | 0.90 | 0.80 | 0.84 |
| Scratch | 0.56 | 0.16 | 0.25 |
| **전체 정확도** | | | **0.85** |

### 해상도별 성능 비교
| 해상도 | 정확도 |
|--------|--------|
| 16×16 | 0.822 |
| 32×32 | 0.845 |
| 48×48 | 0.847 |
| 64×64 | 0.846 |

## 주요 결과
- Edge-Ring F1 0.97, Center F1 0.94 → 형태가 뚜렷한 패턴은 높은 성능
- Scratch F1 0.25 → 선형 패턴은 32×32 리사이즈 시 특징 손실
- Loc ↔ Edge-Loc 오분류 208건 → 공간적으로 유사한 패턴 간 혼동
- 해상도 48×48에서 최고 성능, 이후 개선 없음 → RF의 공간 관계 학습 한계

## 한계 및 개선 방향
- Scratch 패턴: CNN 적용 시 선형 특징 보존으로 성능 개선 가능
- Loc/Edge-Loc 혼동: 공간적 위치 특징 추가 엔지니어링 필요
- 해상도 한계: RandomForest는 픽셀 간 공간 관계 학습 불가 → CNN 전환 필요
