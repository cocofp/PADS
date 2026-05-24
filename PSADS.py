#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ADS优化的分区SMOGN数据生成（36特征版本，7:3随机划分）
生成日期: 2025-11-14
================================================================================
目标：从临近3个候选seed中，为每个zone智能优选最佳合成数据（基于 split_zone_data_70_30_20251114.py 的3折训练集）

核心逻辑：
  1. 对每个target_seed，使用其临近3个seeds作为候选池（包括自己+前后各1个）
  2. 分区级别ADS：每个zone独立训练Reference Ensemble，独立优选
  3. 循环结构：Seeds → Splits → Zones → Multiples

改进内容：
  【第一阶段：真实性-一致性筛选（Reference Ensemble版本）】
  1. Reference Ensemble: 3个异质回归模型（KNeighborsRegressor, ElasticNet, LGBMRegressor）
     - 避免与下游预测模型（RFR, XGB, Ridge, MLP）重复，减少模型偏差
  2. 真实性评分A_i: 
     - 使用均值预测: μ_mean = mean(y_hat_KNN, y_hat_ElasticNet, y_hat_LightGBM)
     - 计算残差: r_i = |y_i_syn - μ_mean|
     - 使用标准差标准化: z_i = (r_i - mean(R)) / (std(R) + ε)
     - Sigmoid转换: A_i = 1 / (1 + exp(z_i - τ_A))，其中 τ_A = 2.0, ε = 1e-6
  3. 一致性评分C_i:
     - 计算3个模型预测的方差: σ_i^2 = var(y_hat_KNN, y_hat_ElasticNet, y_hat_LightGBM)
     - 使用全局平均方差归一化: C_i = exp(-σ_i^2 / (mean(Σ^2) + ε))
     - 值越大表示一致性越好（方差越小，一致性越高）
  4. 综合评分: S_i = A_i × C_i
  
  【第二阶段：多样性选择（K-Center贪心算法）】
  5. 双重阈值筛选: 仅保留 A_i > τ_A AND C_i > τ_C 的样本进入C_qual (τ_A=0.5, τ_C=0.5)
  6. 综合评分: S_i = A_i × C_i
  7. 初始样本 s_1: 选择C_qual中综合评分S最高的样本
  8. 迭代选择:
     - 计算最小距离 d_i^min = min_{s∈S} ||o_i - o_s||_2
     - 自适应Top-K选择 K = max(10, min(100, 0.05 × n_remaining))
     - 在Top-K距离最大的候选中，选择综合评分最高的样本 s_t
  9. 最终增强集: D_aug = D_real ∪ D_selected
  
  【其他优化】
  10. 样本分离：ADS只对合成样本筛选，真实样本全部保留
  11. 候选池：临近3个seeds（当前seed+前后各1个）
  12. 内存管理：每个zone处理完后立即保存临时文件，最后合并并清理



  说明：
  - 本脚本中的 Reference Ensemble 仅用于"样本质量评估"（教师委员会），
    使用KNN + ElasticNet + LightGBM，避免与下游预测模型（RFR, XGB, Ridge, MLP）重复。
    下游回归实验中的 RFR / XGB / Ridge / MLP 将在各自的回归训练脚本中独立重新训练，
    功能与训练过程与这里的教师模型相互独立，避免"既当运动员又当裁判"。
================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from lightgbm import LGBMRegressor
import json
import warnings
import time
import sys
import os
import gc
from datetime import datetime
warnings.filterwarnings('ignore')

# 设置Windows控制台UTF-8编码（解决中文乱码问题）
if sys.platform == 'win32':
    try:
        os.system('chcp 65001 >nul 2>&1')
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
        os.environ['PYTHONIOENCODING'] = 'utf-8'
    except Exception:
        pass

print("=" * 100)
print("ADS优化的分区SMOGN数据生成 - 基于36特征（分区版）")
print(f"生成日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 100)

# ============================================================================
# 配置参数
# ============================================================================
ROOT_DIR = Path(r'F:\xf\SCI4\Data augmentation_TL\Data augmentation1')
BASE_DIR = ROOT_DIR / '4.agrizone'
DATA_DIR = BASE_DIR / 'data_random_split_zone_70_30'
# SMOGN输入目录：对应 generate_zone_based_smogn_split70_20251114.py 的输出路径
SMOGN_INPUT_DIR = BASE_DIR / 'zone_based_smogn_split70_20251114'
ADS_OUTPUT_DIR = BASE_DIR / 'script' / 'zone_based_smogn_ads_split70_20251114'
ADS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 进度文件
PROGRESS_FILE = ADS_OUTPUT_DIR / 'ads_generation_progress_split70.json'

TARGET_COL = 'yield'
ZONE_COL = 'AgriZone'

# 36个特征
ALL_36_FEATURES = [
    'gcvi_4', 'gcvi_5', 'gcvi_6', 'gcvi_7', 'gcvi_8', 'gcvi_9',
    'tmax_4', 'tmin_4', 'precip_4', 'rad_4', 'vpd_4',
    'tmax_5', 'tmin_5', 'precip_5', 'rad_5', 'vpd_5',
    'tmax_6', 'tmin_6', 'precip_6', 'rad_6', 'vpd_6',
    'tmax_7', 'tmin_7', 'precip_7', 'rad_7', 'vpd_7',
    'tmax_8', 'tmin_8', 'precip_8', 'rad_8', 'vpd_8',
    'tmax_9', 'tmin_9', 'precip_9', 'rad_9', 'vpd_9'
]

# 快速测试配置
QUICK_TEST_MODE = False  # 快速测试模式开关

if QUICK_TEST_MODE:
    SEEDS = [2222, 9999, 3333]
    MULTIPLES = [2, 3, 4, 5]
    SPLITS = [1]
    print("="*100)
    print("⚠️  快速测试模式已启用")
    print(f"   Seeds: {SEEDS}")
    print(f"   Splits: {SPLITS}")
    print(f"   Multiples: {MULTIPLES}")
    print("="*100)
else:
    SEEDS = [
        2222, 1111, 111, 9999, 1234, 123, 42, 1010, 3333, 5678
    ]
    MULTIPLES = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    SPLITS = [1, 2, 3]
    print("="*100)
    print("✅ 完整模式已启用")
    print(f"   Seeds: {SEEDS}")
    print(f"   Splits: {SPLITS}")
    print(f"   Multiples: {MULTIPLES}")
    print(f"   总计: {len(SEEDS)} × {len(MULTIPLES)} × {len(SPLITS)} = {len(SEEDS) * len(MULTIPLES) * len(SPLITS)} 个组合")
    print("="*100)

FOLDS = []
for split_idx in SPLITS:
    train_file = 'train_zone70.csv' if split_idx == 1 else f'train_zone70_split{split_idx}.csv'
    smogn_suffix = '' if split_idx == 1 else f'_split{split_idx}'
    FOLDS.append({
        'fold': split_idx,
        'split_idx': split_idx,
        'train_file': train_file,
        'smogn_suffix': smogn_suffix,
        'output_subdir': f'split{split_idx}'
    })

print(f"\n配置信息:")
print(f"  数据目录: {DATA_DIR}")
print(f"  SMOGN输入目录: {SMOGN_INPUT_DIR}")
print(f"  ADS输出目录: {ADS_OUTPUT_DIR}")
print(f"  特征数量: {len(ALL_36_FEATURES)}")
print(f"  种子数量: {len(SEEDS)} (种子列表: {SEEDS})")
print(f"  倍数范围: {MULTIPLES}")
print(f"  Splits: {SPLITS}")
print(f"  输出子目录: {[f['output_subdir'] for f in FOLDS]}")
print(f"  模式: {'快速测试模式' if QUICK_TEST_MODE else '完整模式'}")
print(f"  K-Center参数K: 自适应（5%比例，最少10最多100）")
print(f"  双重阈值: τ_A=0.5 (真实性), τ_C=0.5 (一致性) - 筛选准入门槛")
print(f"  综合评分: S = A × C - 在合格样本中排序优选")
print(f"  Reference Ensemble: KNeighborsRegressor + ElasticNet + LGBMRegressor（仅作为教师委员会，避免与下游模型重复）")
print("=" * 100)

# ============================================================================
# 进度保存功能
# ============================================================================
def save_progress(progress_info):
    try:
        with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
            json.dump(progress_info, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"    [警告] 保存进度失败: {e}")

def load_progress():
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"    [警告] 加载进度失败: {e}")
            return None
    return None


def get_train_file(split_idx: int) -> Path:
    if split_idx == 1:
        return DATA_DIR / 'train_zone70.csv'
    return DATA_DIR / f'train_zone70_split{split_idx}.csv'


def get_smogn_file(multiple: int, seed: int, split_idx: int) -> Path:
    if split_idx == 1:
        return SMOGN_INPUT_DIR / f'zone_based_smogn_{multiple}x_seed{seed}_all.csv'
    return SMOGN_INPUT_DIR / f'zone_based_smogn_{multiple}x_seed{seed}_split{split_idx}_all.csv'


def get_output_file(split_idx: int, multiple: int, seed: int) -> Path:
    split_dir = ADS_OUTPUT_DIR / f'split{split_idx}'
    split_dir.mkdir(parents=True, exist_ok=True)
    return split_dir / f'zone_based_smogn_ads_{multiple}x_seed{seed}_split{split_idx}.csv'


def get_temp_file(split_idx: int, zone: str, seed: int, multiple: int) -> Path:
    temp_dir = ADS_OUTPUT_DIR / f'split{split_idx}' / 'temp'
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir / f'temp_split{split_idx}_zone{zone}_seed{seed}_{multiple}x.csv'

# ============================================================================
# 工具函数
# ============================================================================
def get_neighboring_seeds(target_seed, all_seeds, num_neighbors=3):
    """
    获取临近seeds作为候选池：当前seed + 前后各1个（循环）
    """
    if target_seed not in all_seeds:
        raise ValueError(f"Target seed {target_seed} not in seed list")
    
    seed_index = all_seeds.index(target_seed)
    neighbor_indices = [
        (seed_index - 1) % len(all_seeds),
        seed_index,
        (seed_index + 1) % len(all_seeds),
    ]
    return [all_seeds[i] for i in neighbor_indices]

# ============================================================================
# Reference Ensemble（3个异质模型：KNN + ElasticNet + LightGBM）
# ============================================================================
class ReferenceEnsemble:
    """
    Reference Ensemble（参考集成，用作教师委员会）

    使用3个异质回归模型：
    1. KNeighborsRegressor (KNN) - 非参数模型，基于实例
    2. ElasticNet - 线性模型，L1+L2正则化
    3. LGBMRegressor (LightGBM) - 梯度提升树模型

    说明：
    - 这里只负责为ADS计算真实性/一致性分数，不参与下游回归实验。
    - 选择这3个模型是为了避免与下游预测模型（RFR, XGB, Ridge, MLP）重复，减少模型偏差。
    - 下游的RFR/XGB/Ridge/MLP会在回归训练脚本中独立重新训练。
    """
    def __init__(self):
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.is_fitted = False
        self.n_samples = 0

        # KNN：非参数模型，适合小样本
        self.knn = KNeighborsRegressor(
            n_neighbors=5,
            weights='distance',
            metric='euclidean',
            n_jobs=-1,
        )

        # ElasticNet：线性模型，L1+L2正则化
        self.elastic = ElasticNet(
            alpha=1.0,
            l1_ratio=0.5,
            max_iter=1000,
            random_state=42,
        )

        # LightGBM：梯度提升树，与XGB不同实现
        self.lgbm = LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,  # 静默模式
        )

    def fit(self, X, y):
        """训练 KNN + ElasticNet + LightGBM 教师委员会"""
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()
        self.n_samples = len(X_scaled)

        self.knn.fit(X_scaled, y_scaled)
        self.elastic.fit(X_scaled, y_scaled)
        self.lgbm.fit(X_scaled, y_scaled)

        self.is_fitted = True
    
    def predict_ensemble(self, X):
        """返回 KNN、ElasticNet、LightGBM 的预测值（已反标准化到原始y尺度）"""
        if not self.is_fitted:
            raise ValueError("ReferenceEnsemble 模型未训练！")
        
        X_scaled = self.scaler_X.transform(X)
        pred_knn_scaled = self.knn.predict(X_scaled)
        pred_elastic_scaled = self.elastic.predict(X_scaled)
        pred_lgbm_scaled = self.lgbm.predict(X_scaled)

        pred_knn = self.scaler_y.inverse_transform(
            pred_knn_scaled.reshape(-1, 1)
        ).ravel()
        pred_elastic = self.scaler_y.inverse_transform(
            pred_elastic_scaled.reshape(-1, 1)
        ).ravel()
        pred_lgbm = self.scaler_y.inverse_transform(
            pred_lgbm_scaled.reshape(-1, 1)
        ).ravel()
        
        return pred_knn, pred_elastic, pred_lgbm
    
    def compute_AC_scores(self, X, y_true):
        """
        计算A分数（真实性）和C分数（一致性）

        1. 真实性 A_i：
           - μ_mean = mean(y_hat_KNN, y_hat_ElasticNet, y_hat_LightGBM)
           - r_i = |y_i - μ_mean|
           - z_i = (r_i - mean(R)) / (std(R) + ε)
           - A_i = 1 / (1 + exp(z_i - τ_A)), τ_A = 2.0

        2. 一致性 C_i：
           - 对3个预测计算方差：σ_i^2 = var(y_hat_KNN, y_hat_ElasticNet, y_hat_LightGBM)
           - 全局平均方差: mean(Σ^2)
           - C_i = exp(-σ_i^2 / (mean(Σ^2) + ε))
        """
        pred_knn, pred_elastic, pred_lgbm = self.predict_ensemble(X)
        predictions = np.column_stack([pred_knn, pred_elastic, pred_lgbm])

        # --- 真实性 A ---
        mu_mean = predictions.mean(axis=1)
        residuals = np.abs(y_true - mu_mean)

        epsilon = 1e-6
        mean_residual = residuals.mean()
        std_residual = residuals.std()
        z_res = (residuals - mean_residual) / (std_residual + epsilon)

        tau_A = 2.0
        A = 1.0 / (1.0 + np.exp(z_res - tau_A))

        # --- 一致性 C ---
        sigma2 = np.var(predictions, axis=1)          # 每个样本的预测方差 σ_i^2
        sigma2_mean = sigma2.mean()                   # 全局平均方差 mean(Σ^2)
        C = np.exp(-sigma2 / (sigma2_mean + epsilon))

        return A, C, mu_mean


# ============================================================================
# 改进的ADS Selector
# ============================================================================
class ImprovedADSSelector:
    """
    改进的ADS选择器（向量化 + K-center贪心）

    1. 第一阶段：真实性-一致性筛选
       - 用 ReferenceEnsemble 计算 A、C、S=A×C
    2. 第二阶段：多样性（K-center贪心） + 双重阈值
    """
    
    def __init__(self, reference_ensemble, feature_cols, tau_A=0.5, tau_C=0.5):
        self.reference_ensemble = reference_ensemble
        self.feature_cols = feature_cols
        self.tau_A = tau_A
        self.tau_C = tau_C
        self.scaler_feat = StandardScaler()
    
    def get_adaptive_top_k(self, n_remaining):
        """自适应top-K：5%比例，限制在[10, 100]"""
        top_k = max(10, min(100, int(n_remaining * 0.05)))
        return top_k
    
    def select_by_kcenter_and_score(self, candidate_pool, target_n, 
                                     y_col='yield', verbose=False):
        if verbose:
            print(f"      ADS选择：候选池={len(candidate_pool)}，目标={target_n}")
        
        # 计算 A / C / S
        A, C, pred = self.reference_ensemble.compute_AC_scores(
            candidate_pool[self.feature_cols],
            candidate_pool[y_col]
        )
        S = A * C
        
        # 双重阈值
        qual_mask = (A > self.tau_A) & (C > self.tau_C)
        if qual_mask.sum() == 0:
            n_keep = max(target_n, len(candidate_pool) // 2)
            qual_indices = np.argsort(S)[-n_keep:]
            if verbose:
                print(f"      [警告] 无样本满足 A>{self.tau_A}, C>{self.tau_C}，使用前{n_keep}个高分样本")
        else:
            qual_indices = np.where(qual_mask)[0]
            if verbose:
                print(f"      双重阈值筛选: {len(candidate_pool)} → {len(qual_indices)}")
        
        C_qual = candidate_pool.iloc[qual_indices].copy().reset_index(drop=True)
        A_qual = np.array(A[qual_indices], dtype=np.float64)
        C_qual_scores = np.array(C[qual_indices], dtype=np.float64)
        S_qual = np.array(S[qual_indices], dtype=np.float64)
        
        if len(C_qual) <= target_n:
            if verbose:
                print(f"      高质量样本数 ({len(C_qual)}) ≤ 目标数 ({target_n})，直接返回")
            C_qual['A_score'] = A_qual
            C_qual['C_score'] = C_qual_scores
            C_qual['S_score'] = S_qual
            return C_qual
        
        # K-center 贪心
        feat_scaled = self.scaler_feat.fit_transform(
            C_qual[self.feature_cols]
        )
        
        selected_indices = []
        remaining = list(range(len(C_qual)))
        
        first_idx = int(np.argmax(S_qual))
        selected_indices.append(first_idx)
        remaining.remove(first_idx)
        
        if verbose:
            print(f"      第1个样本: idx={first_idx}, S={S_qual[first_idx]:.3f}")
        
        for i in range(target_n - 1):
            if len(remaining) == 0:
                break
            
            dists = cdist(
                feat_scaled[remaining],
                feat_scaled[selected_indices],
                metric='euclidean'
            )
            min_dists = dists.min(axis=1)
            
            k = self.get_adaptive_top_k(len(remaining))
            if k > len(min_dists):
                k = len(min_dists)
            if k == 0:
                k = 1
            top_dist_local_indices = np.argsort(min_dists)[-k:]
            
            remaining_arr = np.array(remaining, dtype=np.int64)
            top_dist_global_indices = remaining_arr[top_dist_local_indices]
            
            if len(top_dist_global_indices) == 0:
                best_local_idx = np.argmax(min_dists)
                best_idx = remaining[best_local_idx]
            else:
                valid_indices = top_dist_global_indices[
                    (top_dist_global_indices >= 0) &
                    (top_dist_global_indices < len(S_qual))
                ]
                if len(valid_indices) == 0:
                    best_idx = remaining[0] if len(remaining) > 0 else 0
                else:
                    S_qual_topk = S_qual[valid_indices]
                    best_local_idx = np.argmax(S_qual_topk)
                    best_idx = valid_indices[best_local_idx]
            
            selected_indices.append(best_idx)
            remaining.remove(best_idx)
            
            if verbose and (i + 1) % 200 == 0:
                print(f"      已选择: {i + 1}/{target_n - 1}")
        
        # 去重 + 边界检查
        selected_indices = list(selected_indices)
        seen = set()
        selected_indices_unique = []
        for idx in selected_indices:
            if idx not in seen and 0 <= idx < len(C_qual):
                seen.add(idx)
                selected_indices_unique.append(idx)
        
        if len(selected_indices_unique) < target_n and len(selected_indices_unique) < len(C_qual):
            remaining_indices = [i for i in range(len(C_qual)) if i not in selected_indices_unique]
            if len(remaining_indices) > 0:
                remaining_scores = S_qual[remaining_indices]
                needed = min(target_n - len(selected_indices_unique), len(remaining_indices))
                top_remaining = np.argsort(remaining_scores)[-needed:]
                selected_indices_unique.extend([remaining_indices[i] for i in top_remaining])
        
        selected_indices_array = np.array(selected_indices_unique, dtype=np.int64)
        
        refined = C_qual.iloc[selected_indices_array].copy()
        refined['A_score'] = A_qual[selected_indices_array]
        refined['C_score'] = C_qual_scores[selected_indices_array]
        refined['S_score'] = S_qual[selected_indices_array]
        
        if verbose:
            print(f"      完成！A均值={refined['A_score'].mean():.3f}, "
                  f"C均值={refined['C_score'].mean():.3f}, "
                  f"S均值={refined['S_score'].mean():.3f}")
        
        return refined

# ============================================================================
# 主循环：Seeds → Splits → Zones → Multiples
# ============================================================================
print(f"\n{'=' * 100}")
print(f"开始ADS优选数据生成")
print(f"{'=' * 100}\n")

start_time = time.time()
total_tasks = len(SEEDS) * len(FOLDS) * len(MULTIPLES)
completed = 0
skipped = 0

# 加载已有进度
saved_progress = load_progress()
if saved_progress:
    print(f"\n[断点继续] 检测到已有进度文件")
    print(f"  上次完成: {saved_progress.get('last_completed_seed', 'N/A')}")
    print(f"  上次完成Split: {saved_progress.get('last_completed_split', 'N/A')}")
    print(f"  已完成任务: {saved_progress.get('completed', 0)}/{total_tasks}")
else:
    print(f"\n[新任务] 从头开始")

for seed_idx, target_seed in enumerate(SEEDS, 1):
    print(f"\n{'#' * 100}")
    print(f"[{seed_idx}/{len(SEEDS)}] Target Seed: {target_seed}")
    print(f"{'#' * 100}")
    
    candidate_seeds = get_neighboring_seeds(target_seed, SEEDS)
    print(f"  候选Seeds: {candidate_seeds}")
    
    for fold_info in FOLDS:
        fold_idx = fold_info['fold']
        split_idx = fold_info['split_idx']
        
        print(f"\n  {'=' * 95}")
        print(f"  Split {split_idx}")
        print(f"  {'=' * 95}")
        
        train_real = pd.read_csv(DATA_DIR / fold_info['train_file'])
        
        required_cols = ALL_36_FEATURES + [TARGET_COL, ZONE_COL]
        missing_cols = [col for col in required_cols if col not in train_real.columns]
        if missing_cols:
            print(f"    [警告] 缺少列: {missing_cols}，跳过该split")
            continue
        
        keep_cols = ALL_36_FEATURES + [TARGET_COL, ZONE_COL]
        optional_cols = ['year1', 'City']
        for col in optional_cols:
            if col in train_real.columns:
                keep_cols.append(col)
        
        train_real = train_real[keep_cols].copy()
        zones = sorted(train_real[ZONE_COL].unique())
        
        print(f"    真实训练集: {len(train_real)} 样本")
        print(f"    分区数量: {len(zones)} ({zones})")
        
        output_fold_dir = ADS_OUTPUT_DIR / fold_info['output_subdir']
        output_fold_dir.mkdir(parents=True, exist_ok=True)
        smogn_fold_dir = SMOGN_INPUT_DIR
        
        for mult_idx, multiple in enumerate(MULTIPLES, 1):
            output_file = output_fold_dir / f'zone_based_smogn_ads_{multiple}x_seed{target_seed}_split{split_idx}.csv'
            
            if output_file.exists():
                file_size = output_file.stat().st_size
                if file_size > 0:
                    print(f"    [{mult_idx}/{len(MULTIPLES)}] {multiple}x: 已存在（{file_size/1024:.1f}KB），跳过")
                    skipped += 1
                    completed += 1
                    continue
            
            print(f"\n    [{mult_idx}/{len(MULTIPLES)}] {multiple}x:")
            task_start = time.time()
            
            temp_file_pattern = f'temp_split{split_idx}_zone*_seed{target_seed}_{multiple}x.csv'
            existing_temp_files = list(output_fold_dir.glob(temp_file_pattern))
            completed_zones = set()
            
            if existing_temp_files:
                for temp_file in existing_temp_files:
                    try:
                        zone_name = temp_file.stem.split('_seed')[0].replace('temp_zone', '')
                        if temp_file.exists() and temp_file.stat().st_size > 0:
                            completed_zones.add(zone_name)
                    except Exception:
                        pass
                
                if completed_zones:
                    print(f"      发现 {len(completed_zones)} 个已完成的zone临时文件，将利用它们恢复进度")
            
            temp_files = []
            for temp_file in existing_temp_files:
                if temp_file.exists() and temp_file.stat().st_size > 0:
                    temp_files.append(temp_file)
            
            for zone_idx, zone in enumerate(zones, 1):
                print(f"      [{zone_idx}/{len(zones)}] 分区: {zone}", end=' | ', flush=True)
                zone_start = time.time()
                
                zone_temp_file = output_fold_dir / f'temp_zone{zone}_seed{target_seed}_{multiple}x.csv'
                if zone_temp_file.exists() and zone_temp_file.stat().st_size > 0:
                    if zone_temp_file not in temp_files:
                        temp_files.append(zone_temp_file)
                    print(f"已存在，跳过 | {zone_temp_file.stat().st_size/1024:.1f}KB")
                    continue
                
                train_real_zone = train_real[train_real[ZONE_COL] == zone].copy()
                
                reference_ensemble = ReferenceEnsemble()
                reference_ensemble.fit(train_real_zone[ALL_36_FEATURES], train_real_zone[TARGET_COL])
                
                candidate_pool = []
                real_samples_zone = None
                candidate_pool_data = None
                selector = None
                try:
                    smogn_suffix = fold_info['smogn_suffix']
                    for idx, cand_seed in enumerate(candidate_seeds):
                        smogn_file = smogn_fold_dir / f'zone_based_smogn_{multiple}x_seed{cand_seed}{smogn_suffix}_all.csv'
                        if smogn_file.exists():
                            synth = pd.read_csv(smogn_file)
                            synth_zone = synth[synth[ZONE_COL] == zone].copy()
                            
                            if len(synth_zone) > 0:
                                if 'is_synthetic' in synth_zone.columns:
                                    real_part = synth_zone[synth_zone['is_synthetic'] == 0].copy()
                                    synthetic_part = synth_zone[synth_zone['is_synthetic'] == 1].copy()
                                    
                                    if idx == 0 and len(real_part) > 0:
                                        keep_cols_zone = ALL_36_FEATURES + [TARGET_COL, ZONE_COL, 'is_synthetic']
                                        for col in ['year1', 'City']:
                                            if col in real_part.columns:
                                                keep_cols_zone.append(col)
                                        real_samples_zone = real_part[keep_cols_zone].copy()
                                    
                                    if len(synthetic_part) > 0:
                                        keep_cols_zone = ALL_36_FEATURES + [TARGET_COL, ZONE_COL, 'is_synthetic']
                                        for col in ['year1', 'City']:
                                            if col in synthetic_part.columns:
                                                keep_cols_zone.append(col)
                                        synthetic_part_filtered = synthetic_part[keep_cols_zone].copy()
                                        synthetic_part_filtered['source_seed'] = cand_seed
                                        candidate_pool.append(synthetic_part_filtered)
                                else:
                                    synth_zone['source_seed'] = cand_seed
                                    candidate_pool.append(synth_zone)
                            
                            del synth, synth_zone
                            gc.collect()
                    
                    if len(candidate_pool) == 0:
                        print(f"无合成样本候选数据")
                        del train_real_zone
                        gc.collect()
                        continue
                    
                    if len(candidate_pool) > 0:
                        all_cols = set()
                        for df in candidate_pool:
                            all_cols.update(df.columns)
                        
                        candidate_pool_aligned = []
                        for df in candidate_pool:
                            df_copy = df.copy().reset_index(drop=True)
                            df_aligned = df_copy.reindex(columns=list(all_cols))
                            candidate_pool_aligned.append(df_aligned)
                        
                        candidate_pool_data = pd.concat(candidate_pool_aligned, ignore_index=True)
                        candidate_pool_data = candidate_pool_data.reset_index(drop=True)
                        del candidate_pool, candidate_pool_aligned
                    else:
                        candidate_pool_data = pd.DataFrame(columns=ALL_36_FEATURES + [TARGET_COL, ZONE_COL])
                    gc.collect()
                    
                    target_n = len(train_real_zone) * multiple
                    selector = ImprovedADSSelector(
                        reference_ensemble=reference_ensemble,
                        feature_cols=ALL_36_FEATURES,
                        tau_A=0.5,
                        tau_C=0.5,
                    )
                    
                    refined_data = selector.select_by_kcenter_and_score(
                        candidate_pool_data, 
                        target_n, 
                        y_col=TARGET_COL,
                        verbose=False
                    )
                    
                    cols_to_drop = []
                    for col in ['A_score', 'C_score', 'S_score', 'source_seed']:
                        if col in refined_data.columns:
                            cols_to_drop.append(col)
                    if cols_to_drop:
                        refined_clean = refined_data.drop(columns=cols_to_drop)
                    else:
                        refined_clean = refined_data.copy()
                    
                    if real_samples_zone is not None and len(real_samples_zone) > 0:
                        if 'is_synthetic' not in real_samples_zone.columns:
                            real_samples_zone['is_synthetic'] = 0
                        if 'is_synthetic' not in refined_clean.columns:
                            refined_clean['is_synthetic'] = 1
                        zone_final = pd.concat([real_samples_zone, refined_clean], ignore_index=True)
                    else:
                        if 'is_synthetic' not in refined_clean.columns:
                            refined_clean['is_synthetic'] = 1
                        zone_final = refined_clean
                    
                    temp_file = output_fold_dir / f'temp_zone{zone}_seed{target_seed}_{multiple}x.csv'
                    zone_final.to_csv(temp_file, index=False)
                    if temp_file not in temp_files:
                        temp_files.append(temp_file)
                    
                    real_count = len(real_samples_zone) if real_samples_zone is not None else 0
                    synthetic_count = len(refined_data)
                    print(f"[OK] 真实={real_count} + 合成={synthetic_count}条 | "
                          f"A={refined_data['A_score'].mean():.3f} "
                          f"C={refined_data['C_score'].mean():.3f} | "
                          f"{time.time() - zone_start:.1f}s")
                    
                    del refined_data, refined_clean, zone_final, candidate_pool_data, selector, reference_ensemble, train_real_zone
                    if real_samples_zone is not None:
                        del real_samples_zone
                    gc.collect()
                
                except Exception as e:
                    print(f"[错误] {e}")
                    import traceback
                    traceback.print_exc()
                    for var in ['reference_ensemble', 'selector', 'candidate_pool_data',
                                'train_real_zone', 'candidate_pool']:
                        try:
                            del globals()[var]
                        except Exception:
                            pass
                    gc.collect()
                    continue
            
            if len(temp_files) > 0:
                try:
                    valid_files = [f for f in temp_files if f.exists() and f.stat().st_size > 0]
                    
                    if len(valid_files) == 0:
                        print(f"      [警告] 没有有效的临时文件")
                    else:
                        all_zones_data = []
                        batch_size = 3
                        total_rows = 0
                        
                        for i in range(0, len(valid_files), batch_size):
                            batch_files = valid_files[i:i+batch_size]
                            batch_data_list = []
                            
                            for temp_file in batch_files:
                                try:
                                    zone_data = pd.read_csv(temp_file)
                                    batch_data_list.append(zone_data)
                                    total_rows += len(zone_data)
                                    temp_file.unlink()
                                except Exception as e:
                                    print(f"      [警告] 读取临时文件 {temp_file.name} 失败: {e}")
                            
                            if len(batch_data_list) > 0:
                                all_zones_data.extend(batch_data_list)
                            
                            del batch_data_list
                            gc.collect()
                        
                        if len(all_zones_data) > 0:
                            final_merged = pd.concat(all_zones_data, ignore_index=True)
                            final_merged.to_csv(output_file, index=False)
                            
                            del all_zones_data, final_merged
                            gc.collect()
                            
                            task_time = time.time() - task_start
                            completed += 1
                            
                            output_size = output_file.stat().st_size if output_file.exists() else 0
                            print(f"      >> 合并保存: {len(valid_files)}个分区 | {total_rows}行 | "
                                  f"{output_size/1024:.1f}KB | {task_time:.1f}s | {output_file.name}")
                        else:
                            print(f"      [警告] 没有可合并的数据")
                        
                except Exception as e:
                    print(f"      [错误] 合并失败: {e}")
                    import traceback
                    traceback.print_exc()
                    try:
                        del all_zones_data
                    except Exception:
                        pass
                    try:
                        del final_merged
                    except Exception:
                        pass
                    gc.collect()
            else:
                print(f"      [警告] 没有可保存的数据")
        
        del train_real
        gc.collect()
        
        elapsed = time.time() - start_time
        avg_time = elapsed / max(completed, 1)
        remaining_tasks = total_tasks - completed
        eta = remaining_tasks * avg_time
        print(f"\n    总进度: {completed}/{total_tasks} ({completed/total_tasks*100:.1f}%) | "
              f"跳过: {skipped} | 预计剩余: {eta/60:.1f}分钟")
        
        progress_info = {
            'last_completed_seed': target_seed,
            'last_completed_split': split_idx,
            'completed': completed,
            'skipped': skipped,
            'total_tasks': total_tasks,
            'progress_percent': completed / total_tasks * 100,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        save_progress(progress_info)

# ============================================================================
# 总结
# ============================================================================
print(f"\n{'=' * 100}")
print(f"ADS优选数据生成完成")
print(f"{'=' * 100}\n")

total_time = (time.time() - start_time) / 60
print(f"总耗时: {total_time:.1f}分钟")
print(f"完成任务: {completed}/{total_tasks}")
print(f"跳过任务: {skipped}/{total_tasks}")
print(f"输出目录: {ADS_OUTPUT_DIR}")

final_progress = {
    'status': 'completed',
    'completed': completed,
    'skipped': skipped,
    'total_tasks': total_tasks,
    'progress_percent': 100.0,
    'total_time_minutes': total_time,
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}
save_progress(final_progress)

print(f"\n改进内容总结:")
print(f"  === 第一阶段：真实性-一致性筛选（Reference Ensemble版本） ===")
print(f"  1. Reference Ensemble: 3个异质回归模型（KNN + ElasticNet + LightGBM），仅用作教师委员会")
print(f"     - 避免与下游预测模型（RFR, XGB, Ridge, MLP）重复，减少模型偏差")
print(f"  2. 真实性评分A_i: 使用3模型均值预测 + 标准差标准化 + Sigmoid转换")
print(f"  3. 一致性评分C_i: 使用3模型预测方差 + 全局平均方差归一化 + exp(-σ²/mean(σ²))")
print(f"  4. 综合评分: S_i = A_i × C_i")
print(f"\n  === 第二阶段：多样性选择（K-Center贪心算法） ===")
print(f"  5. 双重阈值筛选: A_i>τ_A AND C_i>τ_C → C_qual (τ_A=0.5, τ_C=0.5)")
print(f"  6. 综合评分: S_i = A_i × C_i，用于在候选中排序优选")
print(f"  7. 初始样本s_1: 选择C_qual中综合评分S最高的样本")
print(f"  8. 迭代选择: 距离最大Top-K + 综合评分最高")
print(f"  9. 最终增强集: D_aug = D_real ∪ D_selected")
print(f"\n  === 其他优化 ===")
print(f"  10. 样本分离: ADS只对合成样本筛选，真实样本全部保留")
print(f"  11. 候选池策略: 临近3个seeds（当前+前后各1个）")
print(f"  12. 分区级别ADS: 每个zone独立训练教师委员会并优选")
print(f"  13. 内存管理: 分区写临时文件，最后合并")
print(f"  14. 下游回归模型中的RFR/XGB/Ridge/MLP将单独重新训练，与本脚本中的教师模型相互独立")

print(f"\n{'=' * 100}")
print(f"完成！")
print(f"{'=' * 100}\n")
