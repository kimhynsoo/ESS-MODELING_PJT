# 데이터

학습과 노트북 실행에는 `processed/`의 CSV만 필요하다. 원본 MATLAB 파일(3개, 약 8.3GB)은 용량 때문에 저장소에 넣지 않았다.

## 원본에서 다시 추출하기

[공개 데이터](https://www.kaggle.com/datasets/itshpark/data-driven-prediction-of-battery-cycle)에서 아래 파일을 받아 `data/raw/`에 두고, 저장소 루트에서 `python -m src.preprocess`를 실행한다.

| 배치    | 파일                                                   |  셀 |
| ------- | ------------------------------------------------------ | --: |
| Batch 1 | `2017-05-12_batchdata_updated_struct_errorcorrect.mat` |  46 |
| Batch 2 | `2018-02-20_batchdata_updated_struct_errorcorrect.mat` |  47 |
| Batch 3 | `2018-04-12_batchdata_updated_struct_errorcorrect.mat` |  46 |

## processed/

| 파일               | 내용                                                                                                        |
| ------------------ | ----------------------------------------------------------------------------------------------------------- |
| `cells.csv`        | 셀당 1행. 수명·충전 정책·실험 절차, 100회까지의 피처, 배치 간 비교 점검값, 원논문 노이즈 채널 표시(`noisy`) |
| `capacity.csv.gz`  | 전체 사이클의 방전 용량. EDA의 열화 곡선·Knee 계산에만 사용                                                 |
| `qv_curves.csv.gz` | 10·100회차 방전 곡선 Q(V). 10점 간격으로 줄인 EDA용                                                         |

피처는 모두 100회차까지의 값이다. ΔQ 로그분산은 전압 격자(3.5→2.0V, 1,000점) 전체에서 계산한다. 수명(`life`)이 비어 있는 10셀은 분석에서 제외한다.
