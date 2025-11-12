"""
数据预处理模块 - "Final Data" 工厂

职责：
1. 加载和预处理原始数据
2. 计算所有必需的特征和目标变量
3. 生成标准化的训练/测试数据集

对应报告 Chapter 3 (数据探索性分析)
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import yaml
from pathlib import Path


def load_config(config_path: str = "configs/default.yaml") -> Dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def calculate_weighted_avg_speed(df: pd.DataFrame) -> pd.Series:
    """
    计算流量加权平均速度
    
    使用 FlowForHarmonicSpeedCalc 和 AverageSpeed 列进行加权平均
    对应报告中的速度计算方法
    
    Args:
        df: 包含流量和速度列的DataFrame
        
    Returns:
        加权平均速度 (km/h)
    """
    # 查找所有车道的流量和速度列
    flow_cols = [col for col in df.columns if 'FlowForHarmonicSpeedCalc' in col]
    speed_cols = [col for col in df.columns if 'AverageSpeedLane' in col]
    
    # 确保列数匹配
    if len(flow_cols) != len(speed_cols):
        # 备选方案：使用简单平均
        speed_cols_simple = [col for col in df.columns if 'AverageSpeed' in col]
        if speed_cols_simple:
            return df[speed_cols_simple].mean(axis=1)
        else:
            raise ValueError("无法找到速度列")
    
    # 计算加权平均速度
    total_flow = df[flow_cols].sum(axis=1)
    weighted_speed = 0
    
    for flow_col, speed_col in zip(flow_cols, speed_cols):
        # 避免除以零
        mask = total_flow > 0
        weighted_speed = weighted_speed + (df[flow_col] * df[speed_col])
    
    # 计算最终的加权平均
    result = weighted_speed / total_flow
    
    # 处理无效值：用中位数填充
    result = result.replace([np.inf, -np.inf], np.nan)
    median_speed = result.median()
    result = result.fillna(median_speed)
    
    return result


def create_final_dataset(
    link_id: int,
    precleaned_filepath: str,
    config: Dict,
    weather_filepath: Optional[str] = None
) -> pd.DataFrame:
    """
    创建 Final Dataset - 核心数据处理函数
    
    这个函数实现了您的愿景："输入任何一个LinkID，它都可以生成标准的FinalData"
    
    执行步骤（基于报告 Chapter 3）：
    1. 加载数据并筛选指定路段
    2. 获取静态属性（容量C、长度L）
    3. 计算V（小时流量）
    4. 计算v（平均速度）
    5. 计算y（Ground Truth行程时间）
    6. 计算t_0（自由流行程时间）
    7. 计算核心特征X1（V/C比）
    8. 计算特征X2（HGV份额）- M3
    9. 计算特征X3（时段）- M1
    10. （未来）合并天气数据 - M4
    
    Args:
        link_id: 路段ID（例如 115030402）
        precleaned_filepath: Precleaned数据文件路径
        config: 配置字典
        weather_filepath: 天气数据文件路径（可选）
        
    Returns:
        包含所有特征和目标变量的DataFrame
    """
    
    print(f"正在加载路段 {link_id} 的数据...")
    
    # ========== 步骤1: 加载数据 ==========
    df = pd.read_excel(precleaned_filepath)
    
    # 筛选指定路段
    df = df[df['LinkUID'] == link_id].copy()
    
    if len(df) == 0:
        raise ValueError(f"未找到 LinkID = {link_id} 的数据")
    
    print(f"  找到 {len(df)} 条记录")
    
    # ========== 步骤2: 获取静态属性 ==========
    # 从配置中获取路段信息
    road_key = None
    for key, road_info in config['roads'].items():
        if road_info['link_id'] == link_id:
            road_key = key
            break
    
    if road_key is None:
        raise ValueError(f"配置文件中未找到 LinkID = {link_id}")
    
    C = config['roads'][road_key]['capacity_vph']  # 容量 (vehicles/hour)
    L = config['roads'][road_key]['length_km']      # 长度 (km)
    
    print(f"  容量 C = {C} vph, 长度 L = {L} km")
    
    # ========== 步骤3: 计算 V (小时流量率) ==========
    # 找到所有流量列
    flow_cols = [col for col in df.columns if 'FlowLane' in col and 'Category' in col and 'Value' in col]
    
    if not flow_cols:
        raise ValueError("未找到流量列")
    
    # 计算15分钟总流量
    df['V_15min_Count'] = df[flow_cols].sum(axis=1)
    
    # 转换为小时流量率（对应报告 Table 3.3: q = 4Q）
    df['V_hourly_rate'] = df['V_15min_Count'] * 4
    
    print(f"  计算了小时流量率 (V = 4Q)")
    
    # ========== 步骤4: 计算 v (平均速度) ==========
    df['v_avg_kmh'] = calculate_weighted_avg_speed(df)
    
    print(f"  计算了加权平均速度")
    
    # ========== 步骤5: 计算 y (Ground Truth 行程时间) ==========
    # 对应报告 Eq. 3.2: T = 3.6L/v
    # 注意：这里 L 是 km，v 是 km/h，结果是秒
    df['t_ground_truth'] = (L / df['v_avg_kmh']) * 3600
    
    # 处理异常值（速度过低导致的极大行程时间）
    # 使用99分位数作为上限
    t_99 = df['t_ground_truth'].quantile(0.99)
    df['t_ground_truth'] = df['t_ground_truth'].clip(upper=t_99)
    
    print(f"  计算了真实行程时间 (Ground Truth)")
    
    # ========== 步骤6: 计算 t_0 (自由流行程时间) ==========
    # 使用低流量时的中位数速度估计自由流速度
    volume_threshold = config['free_flow']['volume_threshold']
    free_flow_data = df[df['V_hourly_rate'] < volume_threshold]
    
    if len(free_flow_data) > 0:
        free_flow_speed = free_flow_data['v_avg_kmh'].median()
    else:
        # 如果没有低流量数据，使用所有数据的90分位数速度
        free_flow_speed = df['v_avg_kmh'].quantile(0.90)
    
    t_0 = (L / free_flow_speed) * 3600
    df['t_0'] = t_0
    
    print(f"  自由流速度 = {free_flow_speed:.2f} km/h, t_0 = {t_0:.2f} 秒")
    
    # ========== 步骤7: 计算核心特征 X1 (V/C 比) ==========
    df['V_C_Ratio'] = df['V_hourly_rate'] / C
    
    print(f"  计算了 V/C 比")
    
    # ========== 步骤8: 计算特征 X2 (HGV 份额) - M3 多类别 ==========
    # 根据配置获取HGV类别
    heavy_categories = config['vehicle_categories']['heavy']
    
    # 找到HGV流量列
    hgv_cols = []
    for cat in heavy_categories:
        hgv_cols.extend([col for col in flow_cols if cat in col])
    
    if hgv_cols:
        df['HGV_Count'] = df[hgv_cols].sum(axis=1)
        df['p_H'] = df['HGV_Count'] / df['V_15min_Count'].replace(0, np.nan)
        df['p_H'] = df['p_H'].fillna(0)  # 无流量时HGV份额为0
    else:
        df['p_H'] = 0
        print("  警告：未找到HGV流量列，p_H 设为 0")
    
    print(f"  计算了 HGV 份额 (p_H)")
    
    # ========== 步骤9: 计算特征 X3 (时段) - M1 动态参数 ==========
    # 解析时间戳
    if 'MeasurementStartUTC' in df.columns:
        df['timestamp'] = pd.to_datetime(df['MeasurementStartUTC'])
    elif 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    else:
        # 尝试其他可能的时间列
        time_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
        if time_cols:
            df['timestamp'] = pd.to_datetime(df[time_cols[0]])
        else:
            raise ValueError("未找到时间戳列")
    
    # 提取时间特征
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['week'] = df['timestamp'].dt.isocalendar().week
    
    # 判断是否工作日（周一到周五）
    df['is_weekday'] = (df['day_of_week'] < 5).astype(int)
    
    # 判断是否高峰时段
    morning_peak_start = int(config['time_periods']['morning_peak']['start'].split(':')[0])
    morning_peak_end = int(config['time_periods']['morning_peak']['end'].split(':')[0])
    evening_peak_start = int(config['time_periods']['evening_peak']['start'].split(':')[0])
    evening_peak_end = int(config['time_periods']['evening_peak']['end'].split(':')[0])
    
    df['is_morning_peak'] = ((df['hour'] >= morning_peak_start) & 
                              (df['hour'] < morning_peak_end)).astype(int)
    df['is_evening_peak'] = ((df['hour'] >= evening_peak_start) & 
                              (df['hour'] < evening_peak_end)).astype(int)
    df['is_peak'] = ((df['is_morning_peak'] == 1) | (df['is_evening_peak'] == 1)).astype(int)
    
    print(f"  计算了时段特征 (is_peak, is_weekday)")
    
    # ========== 步骤10: （未来）合并天气数据 - M4 外部因素 ==========
    if weather_filepath and Path(weather_filepath).exists():
        try:
            weather_df = pd.read_csv(weather_filepath)
            weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp'])
            df = pd.merge_asof(df.sort_values('timestamp'), 
                              weather_df.sort_values('timestamp'),
                              on='timestamp',
                              direction='nearest')
            print(f"  合并了天气数据")
        except Exception as e:
            print(f"  警告：无法加载天气数据 - {e}")
            df['is_raining'] = 0
            df['temperature'] = np.nan
    else:
        # 暂时设置默认值
        df['is_raining'] = 0
        df['temperature'] = np.nan
    
    # ========== 数据清洗 ==========
    # 移除异常值
    df = df[df['t_ground_truth'] > 0]
    df = df[df['v_avg_kmh'] > 0]
    df = df[df['V_C_Ratio'] >= 0]
    
    # 移除缺失值
    required_cols = ['t_ground_truth', 't_0', 'V_C_Ratio', 'v_avg_kmh', 'V_hourly_rate']
    df = df.dropna(subset=required_cols)
    
    print(f"  清洗后剩余 {len(df)} 条记录")
    
    # ========== 返回 Final Data ==========
    # 选择最终需要的列
    final_columns = [
        'timestamp', 'week', 'hour', 'day_of_week',
        't_ground_truth',  # y (目标变量)
        't_0',             # 自由流行程时间
        'V_hourly_rate',   # 流量
        'v_avg_kmh',       # 速度
        'V_C_Ratio',       # X1: 核心特征
        'p_H',             # X2: HGV份额 (M3)
        'is_peak',         # X3: 高峰标志 (M1)
        'is_weekday',      # X3: 工作日标志 (M1)
        'is_morning_peak', # X3: 早高峰
        'is_evening_peak', # X3: 晚高峰
        'is_raining',      # X4: 天气 (M4)
        'temperature'      # X4: 温度 (M4)
    ]
    
    df_final = df[final_columns].copy()
    
    print(f"✓ Final Dataset 创建完成！")
    print(f"  形状: {df_final.shape}")
    print(f"  时间范围: {df_final['timestamp'].min()} 到 {df_final['timestamp'].max()}")
    
    return df_final


def split_data(
    df: pd.DataFrame,
    config: Dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    分割训练集和测试集
    
    使用"分块时间分割法"（对应报告中的方法）：
    - 第1-3周：训练集
    - 第4周：测试集
    
    Args:
        df: Final Dataset
        config: 配置字典
        
    Returns:
        (df_train, df_test) 元组
    """
    
    split_config = config['data_split']
    
    if split_config['method'] == 'temporal_block':
        train_weeks = split_config['train_weeks']
        test_week = split_config['test_week']
        
        # 获取数据的起始周
        min_week = df['week'].min()
        
        # 调整周数（相对于数据起始周）
        df['relative_week'] = df['week'] - min_week + 1
        
        # 分割数据
        df_train = df[df['relative_week'].isin(train_weeks)].copy()
        df_test = df[df['relative_week'] == test_week].copy()
        
        print(f"\n数据分割 (分块时间分割法):")
        print(f"  训练集: 第 {train_weeks} 周, {len(df_train)} 条记录")
        print(f"  测试集: 第 {test_week} 周, {len(df_test)} 条记录")
        
        if len(df_test) == 0:
            print(f"  警告：测试集为空！使用最后20%的数据作为测试集")
            split_idx = int(len(df) * 0.8)
            df_train = df.iloc[:split_idx].copy()
            df_test = df.iloc[split_idx:].copy()
        
        return df_train, df_test
    
    else:
        raise ValueError(f"不支持的分割方法: {split_config['method']}")


def load_and_preprocess(config_path: str = "configs/default.yaml") -> Dict[str, pd.DataFrame]:
    """
    加载和预处理所有数据
    
    这是 run_benchmark.py 调用的主入口函数
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        字典，键为路段名称，值为 Final Dataset
    """
    
    print("=" * 60)
    print("开始数据加载和预处理")
    print("=" * 60)
    
    # 加载配置
    config = load_config(config_path)
    
    # 获取数据文件路径
    base_dir = Path(__file__).parent.parent
    precleaned_file = base_dir / config['data']['precleaned_file']
    weather_file = config['data'].get('weather_file')
    
    if weather_file:
        weather_file = base_dir / weather_file
    
    # 处理所有路段
    all_data = {}
    
    for road_key in config['roads_to_test']:
        road_info = config['roads'][road_key]
        link_id = road_info['link_id']
        
        print(f"\n处理路段: {road_key} (LinkID: {link_id})")
        print("-" * 60)
        
        # 创建 Final Dataset
        df_final = create_final_dataset(
            link_id=link_id,
            precleaned_filepath=str(precleaned_file),
            config=config,
            weather_filepath=str(weather_file) if weather_file else None
        )
        
        all_data[road_key] = df_final
    
    print("\n" + "=" * 60)
    print("数据预处理完成！")
    print("=" * 60)
    
    return all_data


def build_finaldata(
    link_id: int,
    precleaned_path: str,
    snapshot_csv_path: Optional[str] = None,
    capacity: int = None,
    link_length_m: float = None,
    month_start: str = None,
    month_end: str = None,
    t0_strategy: str = "min5pct",
    winsor: Tuple[float, float] = (0.01, 0.99)
) -> pd.DataFrame:
    """
    工程化的FinalData构建函数
    
    这是面向工程的公开接口，统一处理任意LinkID的数据
    
    Args:
        link_id: 路段ID
        precleaned_path: Precleaned数据文件路径
        snapshot_csv_path: Snapshot CSV文件路径（可选）
        capacity: 路段容量（veh/hr），如果为None则从数据中获取
        link_length_m: 路段长度（米），如果为None则从数据中获取
        month_start: 起始月份（如"2024-09-01"）
        month_end: 结束月份（如"2024-09-30"）
        t0_strategy: 自由流时间估计策略
            - "min5pct": 使用最低5%速度的倒数
            - "low_volume": 使用低流量时的中位数速度
        winsor: Winsorize异常值的百分位数（下界，上界）
        
    Returns:
        标准化的FinalData DataFrame，包含以下固定列：
        - datetime: 时间戳
        - LinkUID: 路段ID
        - flow_veh_hr: 小时流量（veh/hr）
        - capacity: 容量（veh/hr）
        - link_length_m: 路段长度（米）
        - fused_tt_15min: 融合行程时间（秒）
        - t0_ff: 自由流行程时间（秒）
        - v_over_c: V/C比
        - count_len_cat1..4: 各类别流量计数
        - share_len_cat1..4: 各类别流量份额
        - hgv_share: HGV份额
        - hour: 小时
        - weekday: 星期几
        - daytype: 日类型
        - is_valid: 是否有效记录
        - flag_tt_outlier: 行程时间异常标志
        - fused_tt_15min_winsor: Winsorize后的行程时间
    """
    
    print(f"\n{'='*60}")
    print(f"构建FinalData: LinkID={link_id}")
    print(f"{'='*60}")
    
    # 1. 加载数据
    print(f"\n[1/8] 加载数据...")
    df = pd.read_excel(precleaned_path)
    df = df[df['LinkUID'] == link_id].copy()
    
    if len(df) == 0:
        raise ValueError(f"未找到 LinkID={link_id} 的数据")
    
    print(f"  找到 {len(df)} 条记录")
    
    # 2. 时间筛选和创建datetime
    print(f"\n[2/8] 处理时间戳...")
    
    # Precleaned数据使用MeasurementDateAdjusted和TimePeriod15MinGroup
    if 'MeasurementDateAdjusted' in df.columns and 'TimePeriod15MinGroup' in df.columns:
        # TimePeriod15MinGroup是整数（0-95），代表一天中的96个15分钟时间段
        # 0 = 00:00, 1 = 00:15, 2 = 00:30, ..., 95 = 23:45
        
        df['date'] = pd.to_datetime(df['MeasurementDateAdjusted'])
        
        # 将TimePeriod15MinGroup转换为分钟数，然后添加到日期上
        df['minutes'] = df['TimePeriod15MinGroup'] * 15
        df['datetime'] = df['date'] + pd.to_timedelta(df['minutes'], unit='m')
        
        print(f"  ✓ 创建了 {len(df)} 个时间戳")
        print(f"  时间范围: {df['datetime'].min()} 至 {df['datetime'].max()}")
        
    elif 'MeasurementStartUTC' in df.columns:
        df['datetime'] = pd.to_datetime(df['MeasurementStartUTC'])
    else:
        # 尝试其他可能的时间列
        time_cols = [col for col in df.columns if 'measurementdate' in col.lower() or 'dateadjusted' in col.lower()]
        if time_cols:
            df['datetime'] = pd.to_datetime(df[time_cols[0]])
        else:
            raise ValueError(f"未找到时间列。可用列: {list(df.columns)[:20]}")
    
    # 处理无效时间戳
    invalid_time = df['datetime'].isna().sum()
    if invalid_time > 0:
        print(f"  警告：{invalid_time} 条记录的时间戳无效，将被移除")
        df = df[df['datetime'].notna()].copy()
    
    # 时间筛选
    if month_start or month_end:
        print(f"  时间筛选...")
        if month_start:
            df = df[df['datetime'] >= month_start]
        if month_end:
            df = df[df['datetime'] <= month_end]
        
        print(f"  筛选后剩余 {len(df)} 条记录")
    
    # 3. 获取路段参数
    print(f"\n[3/8] 获取路段参数...")
    if capacity is None:
        if 'MidPointStaticCapacity' in df.columns:
            capacity = int(df['MidPointStaticCapacity'].iloc[0])
        else:
            raise ValueError("未提供capacity且数据中无MidPointStaticCapacity列")
    
    if link_length_m is None:
        if 'LinkLength' in df.columns:
            link_length_m = float(df['LinkLength'].iloc[0])
        else:
            raise ValueError("未提供link_length_m且数据中无LinkLength列")
    
    df['capacity'] = capacity
    df['link_length_m'] = link_length_m
    df['LinkUID'] = link_id
    
    print(f"  容量: {capacity} veh/hr")
    print(f"  长度: {link_length_m} m ({link_length_m/1000:.3f} km)")
    
    # 4. 计算流量
    print(f"\n[4/8] 计算流量...")
    flow_cols = [col for col in df.columns if 'FlowLane' in col and 'Category' in col and 'Value' in col]
    
    if not flow_cols:
        raise ValueError("未找到流量列")
    
    # 按类别分组计算
    cat1_cols = [col for col in flow_cols if 'Category1' in col]
    cat2_cols = [col for col in flow_cols if 'Category2' in col]
    cat3_cols = [col for col in flow_cols if 'Category3' in col]
    cat4_cols = [col for col in flow_cols if 'Category4' in col]
    
    df['count_len_cat1'] = df[cat1_cols].sum(axis=1) if cat1_cols else 0
    df['count_len_cat2'] = df[cat2_cols].sum(axis=1) if cat2_cols else 0
    df['count_len_cat3'] = df[cat3_cols].sum(axis=1) if cat3_cols else 0
    df['count_len_cat4'] = df[cat4_cols].sum(axis=1) if cat4_cols else 0
    
    # 总流量（15分钟）
    df['V_15min_Count'] = df[flow_cols].sum(axis=1)
    
    # 小时流量率
    df['flow_veh_hr'] = df['V_15min_Count'] * 4
    
    # 计算份额
    total_count = df['V_15min_Count'].replace(0, np.nan)
    df['share_len_cat1'] = df['count_len_cat1'] / total_count
    df['share_len_cat2'] = df['count_len_cat2'] / total_count
    df['share_len_cat3'] = df['count_len_cat3'] / total_count
    df['share_len_cat4'] = df['count_len_cat4'] / total_count
    
    # HGV份额（Category 3和4）
    df['hgv_share'] = (df['count_len_cat3'] + df['count_len_cat4']) / total_count
    df['hgv_share'] = df['hgv_share'].fillna(0)
    
    # V/C比
    df['v_over_c'] = df['flow_veh_hr'] / capacity
    
    print(f"  平均流量: {df['flow_veh_hr'].mean():.0f} veh/hr")
    print(f"  平均V/C: {df['v_over_c'].mean():.3f}")
    print(f"  平均HGV份额: {df['hgv_share'].mean():.3f}")
    
    # 5. 计算速度和行程时间
    print(f"\n[5/8] 计算速度和行程时间...")
    df['v_avg_kmh'] = calculate_weighted_avg_speed(df)
    
    # Fused行程时间（Ground Truth）
    # 优先使用CSV文件中的"Fused Travel Time"（如果提供）
    # 重要：CSV中的Fused Travel Time是秒级/分钟级的真实测量值，需要聚合到15分钟窗口
    if snapshot_csv_path and Path(snapshot_csv_path).exists():
        print(f"  从CSV文件读取Fused Travel Time: {snapshot_csv_path}")
        try:
            df_csv = pd.read_csv(snapshot_csv_path)
            
            # 创建时间戳用于对齐
            # 注意：CSV文件的列名可能有前导空格，pandas读取时会自动处理
            # 但我们需要找到正确的列名
            
            # 查找日期和时间列（处理可能的空格）
            date_col = None
            time_col = None
            
            for col in df_csv.columns:
                col_clean = col.strip().lower()
                if 'local date' in col_clean or ('date' in col_clean and 'time' not in col_clean):
                    date_col = col
                if 'local time' in col_clean or (col_clean == 'time' or ' time' in col.lower()):
                    time_col = col
            
            if not date_col or not time_col:
                # 如果没找到，尝试更宽松的匹配
                for col in df_csv.columns:
                    if 'date' in col.lower() and 'time' not in col.lower():
                        date_col = col
                    if 'time' in col.lower() and 'date' not in col.lower() and 'travel' not in col.lower():
                        time_col = col
            
            if date_col and time_col:
                # 清理数据中的空格
                df_csv[date_col] = df_csv[date_col].astype(str).str.strip()
                df_csv[time_col] = df_csv[time_col].astype(str).str.strip()
                
                # 组合日期和时间
                df_csv['datetime_csv'] = pd.to_datetime(
                    df_csv[date_col] + ' ' + df_csv[time_col],
                    format='%Y-%m-%d %H:%M:%S',
                    errors='coerce'
                )
                
                # 检查是否有无效的时间戳
                invalid_count = df_csv['datetime_csv'].isna().sum()
                if invalid_count > 0:
                    print(f"  警告：{invalid_count} 条记录的时间戳无效，将被忽略")
                    df_csv = df_csv[df_csv['datetime_csv'].notna()].copy()
                
                if len(df_csv) == 0:
                    raise ValueError("所有时间戳都无效")
            else:
                raise ValueError(f"CSV文件缺少日期时间列。找到的列: {list(df_csv.columns)[:10]}")
            
            # 将秒级数据聚合到15分钟窗口
            # 方法：按15分钟窗口分组，计算每个窗口内Fused Travel Time的平均值
            
            # 查找Fused Travel Time列（可能有空格）
            fused_tt_col = None
            for col in df_csv.columns:
                if 'Fused Travel Time' in col or ('fused' in col.lower() and 'travel' in col.lower() and 'time' in col.lower()):
                    fused_tt_col = col
                    break
            
            if not fused_tt_col:
                raise ValueError(f"未找到Fused Travel Time列。可用列: {list(df_csv.columns)}")
            
            df_csv['datetime_15min'] = df_csv['datetime_csv'].dt.floor('15min')
            
            # 聚合：计算每个15分钟窗口的平均Fused Travel Time
            df_csv_agg = df_csv.groupby('datetime_15min')[fused_tt_col].mean().reset_index()
            df_csv_agg.columns = ['datetime_15min', 'fused_tt_15min_from_csv']
            
            print(f"  CSV数据聚合：{len(df_csv)} 条秒级记录 → {len(df_csv_agg)} 个15分钟窗口")
            
            # 对齐到Precleaned数据的15分钟时间戳
            df_sorted = df.sort_values('datetime')
            
            # 使用merge进行精确匹配（15分钟窗口对齐）
            df_merged = pd.merge(
                df_sorted,
                df_csv_agg,
                left_on='datetime',
                right_on='datetime_15min',
                how='left'
            )
            
            # 使用CSV中聚合后的Fused Travel Time
            df['fused_tt_15min'] = df_merged['fused_tt_15min_from_csv'].values
            
            # 检查匹配率
            matched = df['fused_tt_15min'].notna().sum()
            print(f"  ✓ 成功匹配 {matched}/{len(df)} 条记录 ({matched/len(df)*100:.1f}%)")
            
            # 如果匹配率太低，回退到计算值
            if matched < len(df) * 0.5:
                print(f"  警告：匹配率较低 ({matched/len(df)*100:.1f}%)，使用计算值")
                link_length_km = link_length_m / 1000
                calculated_tt = (link_length_km / df['v_avg_kmh']) * 3600
                df['fused_tt_15min'] = df['fused_tt_15min'].fillna(calculated_tt)
            elif matched < len(df):
                # 部分匹配：用计算值填充缺失值
                print(f"  部分匹配：用计算值填充 {len(df) - matched} 个缺失值")
                link_length_km = link_length_m / 1000
                calculated_tt = (link_length_km / df['v_avg_kmh']) * 3600
                df['fused_tt_15min'] = df['fused_tt_15min'].fillna(calculated_tt)
        except Exception as e:
            print(f"  警告：无法从CSV读取Fused Travel Time ({e})，使用计算值")
            import traceback
            traceback.print_exc()
            link_length_km = link_length_m / 1000
            df['fused_tt_15min'] = (link_length_km / df['v_avg_kmh']) * 3600
    else:
        # 如果没有提供CSV或CSV不存在，使用计算值
        print(f"  使用计算值（从速度和长度计算）")
        print(f"  注意：建议提供snapshot_csv_path以使用真实的Fused Travel Time")
        link_length_km = link_length_m / 1000
        df['fused_tt_15min'] = (link_length_km / df['v_avg_kmh']) * 3600
    
    # 处理异常值
    df['fused_tt_15min'] = df['fused_tt_15min'].replace([np.inf, -np.inf], np.nan)
    
    print(f"  平均速度: {df['v_avg_kmh'].mean():.2f} km/h")
    print(f"  平均行程时间: {df['fused_tt_15min'].mean():.2f} 秒")
    
    # 6. 计算自由流行程时间
    print(f"\n[6/8] 计算自由流行程时间 (策略: {t0_strategy})...")
    
    # 重要：T0应该从fused_tt_15min中取最低百分位，而不是从速度计算
    # 这样可以反映真实车辆在完全畅通时的行驶时间
    link_length_km = link_length_m / 1000  # 确保变量可用
    
    if t0_strategy == "min5pct":
        # 取最低5%的fused_tt_15min的均值
        # 逻辑：凌晨或夜间车流极小时，此时的旅行时间可视为自由流条件
        t0_ff = df['fused_tt_15min'].quantile(0.05)
        # 或者使用最低5%的均值（更稳健）
        t0_candidates = df['fused_tt_15min'].nsmallest(int(len(df) * 0.05))
        t0_ff = t0_candidates.mean()
        print(f"  使用最低5%的fused_tt_15min均值")
    elif t0_strategy == "min10pct":
        # 取最低10%的fused_tt_15min的均值
        t0_candidates = df['fused_tt_15min'].nsmallest(int(len(df) * 0.10))
        t0_ff = t0_candidates.mean()
        print(f"  使用最低10%的fused_tt_15min均值")
    elif t0_strategy == "low_volume":
        # 使用低流量时的fused_tt_15min中位数
        low_vol_mask = df['flow_veh_hr'] < 500
        if low_vol_mask.sum() > 10:  # 至少需要10个样本
            t0_ff = df.loc[low_vol_mask, 'fused_tt_15min'].median()
            print(f"  使用低流量(<500 veh/hr)时的fused_tt_15min中位数")
        else:
            # 回退到最低5%
            t0_candidates = df['fused_tt_15min'].nsmallest(int(len(df) * 0.05))
            t0_ff = t0_candidates.mean()
            print(f"  低流量样本不足，使用最低5%的fused_tt_15min均值")
    else:
        raise ValueError(f"未知的t0策略: {t0_strategy}")
    
    # 确保T0合理（不能小于理论最小值）
    theoretical_min = (link_length_km / 120) * 3600  # 假设最大速度120 km/h
    if t0_ff < theoretical_min:
        print(f"  警告：T0 ({t0_ff:.2f}秒) 小于理论最小值 ({theoretical_min:.2f}秒)，使用理论最小值")
        t0_ff = theoretical_min
    
    df['t0_ff'] = t0_ff
    
    print(f"  自由流行程时间 t0: {t0_ff:.2f} 秒")
    print(f"  最低5%范围: [{df['fused_tt_15min'].quantile(0.05):.2f}, {df['fused_tt_15min'].quantile(0.10):.2f}] 秒")
    
    # 7. Winsorize异常值
    print(f"\n[7/8] Winsorize异常值处理...")
    lower_pct, upper_pct = winsor
    lower_bound = df['fused_tt_15min'].quantile(lower_pct)
    upper_bound = df['fused_tt_15min'].quantile(upper_pct)
    
    df['fused_tt_15min_winsor'] = df['fused_tt_15min'].clip(lower=lower_bound, upper=upper_bound)
    df['flag_tt_outlier'] = ((df['fused_tt_15min'] < lower_bound) | 
                              (df['fused_tt_15min'] > upper_bound)).astype(int)
    
    n_outliers = df['flag_tt_outlier'].sum()
    print(f"  异常值数量: {n_outliers} ({n_outliers/len(df)*100:.2f}%)")
    print(f"  Winsor边界: [{lower_bound:.2f}, {upper_bound:.2f}] 秒")
    
    # 8. 时间特征
    print(f"\n[8/8] 提取时间特征...")
    df['hour'] = df['datetime'].dt.hour
    df['weekday'] = df['datetime'].dt.dayofweek
    
    # 日类型（简化版）
    df['daytype'] = df['weekday'].apply(lambda x: 'weekday' if x < 5 else 'weekend')
    
    # 有效性标志
    df['is_valid'] = (
        (df['flow_veh_hr'] > 0) &
        (df['v_avg_kmh'] > 0) &
        (df['fused_tt_15min'].notna()) &
        (df['fused_tt_15min'] > 0)
    ).astype(int)
    
    n_valid = df['is_valid'].sum()
    print(f"  有效记录: {n_valid} / {len(df)} ({n_valid/len(df)*100:.2f}%)")
    
    # 9. 添加时段特征（如果还没有）
    if 'is_peak' not in df.columns:
        if 'hour' in df.columns:
            df['is_peak'] = ((df['hour'] >= 7) & (df['hour'] < 9) |
                            (df['hour'] >= 15) & (df['hour'] < 18)).astype(int)
        else:
            df['is_peak'] = 0
    
    # 添加外部因素（如果还没有）
    if 'is_raining' not in df.columns:
        df['is_raining'] = 0
    if 'temperature' not in df.columns:
        df['temperature'] = np.nan
    
    # 10. 选择最终列（标准FinalData格式）
    final_columns = [
        'datetime', 'LinkUID', 
        'flow_veh_hr', 'capacity', 'link_length_m',
        'fused_tt_15min', 't0_ff', 'v_over_c',
        'count_len_cat1', 'count_len_cat2', 'count_len_cat3', 'count_len_cat4',
        'share_len_cat1', 'share_len_cat2', 'share_len_cat3', 'share_len_cat4',
        'hgv_share',
        'hour', 'weekday', 'daytype',
        'is_peak',  # M1动态参数需要
        'is_raining', 'temperature',  # M4外部因素需要
        'is_valid', 'flag_tt_outlier', 'fused_tt_15min_winsor'
    ]
    
    # 只选择存在的列
    available_columns = [col for col in final_columns if col in df.columns]
    df_final = df[available_columns].copy()
    
    # 检查缺失的列
    missing_columns = [col for col in final_columns if col not in df.columns]
    if missing_columns:
        print(f"  警告：以下列缺失（将使用默认值）: {missing_columns}")
        for col in missing_columns:
            if col == 'is_peak':
                df_final[col] = 0
            elif col in ['is_raining', 'temperature']:
                df_final[col] = 0 if col == 'is_raining' else np.nan
            else:
                df_final[col] = 0
    
    print(f"\n{'='*60}")
    print(f"✓ FinalData构建完成！")
    print(f"  形状: {df_final.shape}")
    print(f"  时间范围: {df_final['datetime'].min()} 至 {df_final['datetime'].max()}")
    print(f"{'='*60}\n")
    
    return df_final


def finaldata_qc_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    生成FinalData的质量控制报告
    
    Args:
        df: FinalData DataFrame
        
    Returns:
        QC报告DataFrame
    """
    
    qc_items = []
    
    # 1. 基本统计
    qc_items.append({
        'Category': 'Basic',
        'Metric': 'Total Records',
        'Value': len(df),
        'Status': 'INFO'
    })
    
    qc_items.append({
        'Category': 'Basic',
        'Metric': 'Valid Records',
        'Value': df['is_valid'].sum(),
        'Status': 'INFO'
    })
    
    qc_items.append({
        'Category': 'Basic',
        'Metric': 'Valid Rate',
        'Value': f"{df['is_valid'].mean()*100:.2f}%",
        'Status': 'PASS' if df['is_valid'].mean() > 0.8 else 'WARNING'
    })
    
    # 2. 流量统计
    qc_items.append({
        'Category': 'Flow',
        'Metric': 'Mean Flow (veh/hr)',
        'Value': f"{df['flow_veh_hr'].mean():.0f}",
        'Status': 'INFO'
    })
    
    qc_items.append({
        'Category': 'Flow',
        'Metric': 'Mean V/C',
        'Value': f"{df['v_over_c'].mean():.3f}",
        'Status': 'INFO'
    })
    
    qc_items.append({
        'Category': 'Flow',
        'Metric': 'Zero Flow Records',
        'Value': (df['flow_veh_hr'] == 0).sum(),
        'Status': 'WARNING' if (df['flow_veh_hr'] == 0).sum() > len(df)*0.1 else 'PASS'
    })
    
    # 3. 行程时间统计
    qc_items.append({
        'Category': 'Travel Time',
        'Metric': 'Mean TT (sec)',
        'Value': f"{df['fused_tt_15min'].mean():.2f}",
        'Status': 'INFO'
    })
    
    qc_items.append({
        'Category': 'Travel Time',
        'Metric': 'Free Flow TT (sec)',
        'Value': f"{df['t0_ff'].iloc[0]:.2f}",
        'Status': 'INFO'
    })
    
    qc_items.append({
        'Category': 'Travel Time',
        'Metric': 'TT Outliers',
        'Value': f"{df['flag_tt_outlier'].sum()} ({df['flag_tt_outlier'].mean()*100:.2f}%)",
        'Status': 'PASS' if df['flag_tt_outlier'].mean() < 0.05 else 'WARNING'
    })
    
    qc_items.append({
        'Category': 'Travel Time',
        'Metric': 'Missing TT',
        'Value': df['fused_tt_15min'].isna().sum(),
        'Status': 'PASS' if df['fused_tt_15min'].isna().sum() == 0 else 'WARNING'
    })
    
    # 4. HGV统计
    qc_items.append({
        'Category': 'HGV',
        'Metric': 'Mean HGV Share',
        'Value': f"{df['hgv_share'].mean()*100:.2f}%",
        'Status': 'INFO'
    })
    
    # 5. 时间覆盖
    time_span = (df['datetime'].max() - df['datetime'].min()).days
    qc_items.append({
        'Category': 'Temporal',
        'Metric': 'Time Span (days)',
        'Value': time_span,
        'Status': 'INFO'
    })
    
    qc_items.append({
        'Category': 'Temporal',
        'Metric': 'Start Date',
        'Value': df['datetime'].min().strftime('%Y-%m-%d'),
        'Status': 'INFO'
    })
    
    qc_items.append({
        'Category': 'Temporal',
        'Metric': 'End Date',
        'Value': df['datetime'].max().strftime('%Y-%m-%d'),
        'Status': 'INFO'
    })
    
    qc_df = pd.DataFrame(qc_items)
    
    return qc_df


if __name__ == "__main__":
    """测试数据加载"""
    
    # 测试新的build_finaldata函数
    print("测试build_finaldata()函数...")
    
    df_final = build_finaldata(
        link_id=115030402,
        precleaned_path="../../Data/Precleaned_M67_Traffic_Data_September_2024.xlsx",
        capacity=6649,
        link_length_m=2713.8037,
        month_start="2024-09-01",
        month_end="2024-09-30",
        t0_strategy="min5pct",
        winsor=(0.01, 0.99)
    )
    
    print("\nFinalData预览:")
    print(df_final.head())
    
    print("\nFinalData统计:")
    print(df_final.describe())
    
    # 生成QC报告
    print("\n生成QC报告...")
    qc_report = finaldata_qc_report(df_final)
    print("\nQC报告:")
    print(qc_report.to_string(index=False))

