"""
问题二：碳感知任务调度模型 —— GPU需求预测
==================================================
数据: workload_trace.xlsx (1200任务, 0~2399h)
策略: 18个序列 (6区域×3任务类型), 各训练一个XGBoost模型
切分: 训练[0-2351] → 验证[2352-2375] → 测试[2376-2399]
"""

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_absolute_error, mean_squared_error

# ============================================================
# 1. 加载数据
# ============================================================
print("=" * 60)
print("Step 1: 加载预处理数据")
print("=" * 60)

with open("/data/workspace/preprocessed_data.pkl", 'rb') as f:
    data = pickle.load(f)

pivot = data['gpu_pivot']
regions = data['regions']
task_types = data['task_types']
max_hour = data['max_hour']

print(f"GPU pivot 形状: {pivot.shape}")
print(f"区域: {regions}")
print(f"任务类型: {task_types}")
print(f"时间范围: 0 ~ {max_hour}")

# ============================================================
# 2. 特征工程
# ============================================================
print("\n" + "=" * 60)
print("Step 2: 特征工程")
print("=" * 60)

def create_features(series, hours_idx, history_window=168):
    """
    从一维时间序列构造监督学习样本。
    返回 X (DataFrame), y (Series), meta [(hour,)]
    """
    X_list, y_list, meta = [], [], []

    for t in range(history_window, len(series)):
        row = []
        # 滞后特征
        for lag in [1, 2, 3, 6, 12, 24, 48, 72, 168]:
            row.append(series[t - lag] if lag <= t else 0.0)
        # 时间特征
        hod = int(hours_idx[t] % 24)
        dow = int((hours_idx[t] // 24) % 7)
        is_we = 1 if dow >= 5 else 0
        row += [hod, dow, is_we]
        # 统计特征 (24h)
        win24 = series[max(0, t-24):t]
        row.append(float(np.mean(win24)) if len(win24) else 0.0)
        row.append(float(np.std(win24))  if len(win24) else 0.0)
        row.append(float(np.max(win24))  if len(win24) else 0.0)
        row.append(float(np.min(win24))  if len(win24) else 0.0)
        # 统计特征 (168h)
        win168 = series[max(0, t-168):t]
        row.append(float(np.mean(win168)) if len(win168) else 0.0)

        X_list.append(row)
        y_list.append(series[t])
        meta.append(hours_idx[t])

    feat_names = ([f'lag_{l}' for l in [1,2,3,6,12,24,48,72,168]]
                  + ['hour_of_day','day_of_week','is_weekend']
                  + ['mean24','std24','max24','min24']
                  + ['mean168'])
    return pd.DataFrame(X_list, columns=feat_names), pd.Series(y_list, name='y'), meta


# ============================================================
# 3. 训练/验证/测试
# ============================================================
print("\n" + "=" * 60)
print("Step 3: 训练 XGBoost 模型")
print("=" * 60)

try:
    import xgboost as xgb
    USE_XGB = True
    print("✅ 使用 XGBoost")
except Exception as e:
    USE_XGB = False
    print(f"⚠️ XGBoost不可用({e}), 回退到 GradientBoosting")
    # 尝试安装
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "xgboost"], check=False)
    try:
        import xgboost as xgb
        USE_XGB = True
        print("✅ xgboost 安装后可用")
    except Exception:
        pass

from sklearn.ensemble import GradientBoostingRegressor

all_preds, all_actual, all_metrics = {}, {}, []

for region in regions:
    for tt in task_types:
        col = (region, tt)
        if col not in pivot.columns:
            print(f"  ⚠️ {col} 不存在, 跳过")
            continue

        series = pivot[col].values.astype(float)
        hours_idx = pivot.index.values

        X, y, meta = create_features(series, hours_idx, history_window=168)

        # 划分
        train_idx = [i for i, m in enumerate(meta) if m <= 2351]
        val_idx   = [i for i, m in enumerate(meta) if 2352 <= m <= 2375]
        test_idx  = [i for i, m in enumerate(meta) if 2376 <= m <= 2399]

        if not train_idx or not test_idx:
            print(f"  ⚠️ {col} 样本不足, 跳过")
            continue

        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx],   y.iloc[val_idx]
        X_te, y_te = X.iloc[test_idx],  y.iloc[test_idx]

        # 训练 (验证集仅用于评估, 不做超参搜索)
        if USE_XGB:
            model = xgb.XGBRegressor(
                n_estimators=200, max_depth=5, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, verbosity=0,
            )
        else:
            model = GradientBoostingRegressor(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                random_state=42,
            )
        model.fit(X_tr, y_tr)

        val_pred = model.predict(X_va)
        val_mae = mean_absolute_error(y_va, val_pred)
        val_rmse = np.sqrt(mean_squared_error(y_va, val_pred))

        # 全量重训 (0~2375)
        full_idx = [i for i, m in enumerate(meta) if m <= 2375]
        if USE_XGB:
            final = xgb.XGBRegressor(
                n_estimators=200, max_depth=5, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, verbosity=0,
            )
        else:
            final = GradientBoostingRegressor(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                random_state=42,
            )
        final.fit(X.iloc[full_idx], y.iloc[full_idx])

        test_pred = final.predict(X_te)
        test_pred = np.maximum(test_pred, 0)

        te_mae  = mean_absolute_error(y_te, test_pred)
        te_rmse = np.sqrt(mean_squared_error(y_te, test_pred))
        mask = y_te.values != 0
        te_mape = (np.mean(np.abs((y_te.values[mask] - test_pred[mask]) / y_te.values[mask])) * 100
                   if mask.sum() else 0.0)

        print(f"  {region:7s}-{tt:18s} | 训练{len(X_tr):5d} | "
              f"验证MAE={val_mae:6.2f} 测试MAE={te_mae:6.2f} "
              f"RMSE={te_rmse:6.2f} MAPE={te_mape:5.1f}%")

        # 存储 —— test_idx 是 X 中的行号，直接用于 test_pred / y_te
        pred_dict = {}
        act_dict  = {}
        for k, i in enumerate(test_idx):
            h = int(meta[i])
            pred_dict[h] = float(test_pred[k])
            act_dict[h]  = float(y_te.values[k])
        all_preds[col] = pred_dict
        all_actual[col] = act_dict
        all_metrics.append({
            'Region': region, 'TaskType': tt,
            'Train_N': len(X_tr), 'Val_MAE': val_mae, 'Val_RMSE': val_rmse,
            'Test_MAE': te_mae, 'Test_RMSE': te_rmse, 'Test_MAPE': te_mape,
        })

# ============================================================
# 4. 汇总
# ============================================================
print("\n" + "=" * 60)
print("Step 4: 汇总")
print("=" * 60)

mdf = pd.DataFrame(all_metrics)
print("\n各序列精度:")
print(mdf[['Region','TaskType','Test_MAE','Test_RMSE','Test_MAPE']].to_string(index=False))
print(f"\n总体平均: MAE={mdf['Test_MAE'].mean():.2f}  "
      f"RMSE={mdf['Test_RMSE'].mean():.2f}  "
      f"MAPE={mdf['Test_MAPE'].mean():.1f}%")

# 保存
pred_df = pd.DataFrame(columns=['Hour','Region','TaskType','Predicted_GPU','Actual_GPU'])
for (r, tt), pd_ in all_preds.items():
    ad = all_actual.get((r, tt), {})
    for h in sorted(pd_.keys()):
        pred_df = pd.concat([pred_df, pd.DataFrame([{
            'Hour': h, 'Region': r, 'TaskType': tt,
            'Predicted_GPU': pd_[h], 'Actual_GPU': ad.get(h, 0),
        }])], ignore_index=True)
pred_df.to_csv("/data/workspace/q2_predictions.csv", index=False)
mdf.to_csv("/data/workspace/q2_metrics.csv", index=False)

with open("/data/workspace/q2_predictions.pkl", 'wb') as f:
    pickle.dump({'predictions': all_preds, 'actuals': all_actual,
                 'metrics': mdf}, f)

print(f"\n✅ 预测结果: {len(pred_df)} 行 → q2_predictions.csv")
print(f"✅ 评估指标 → q2_metrics.csv")
print(f"✅ 字典     → q2_predictions.pkl")
