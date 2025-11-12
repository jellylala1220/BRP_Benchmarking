"""
构建FinalData的CLI工具

使用方法:
    python -m pipelines.build_finaldata --link 115030402 \\
        --preclean "Data/Precleaned_M67_Traffic_Data_September_2024.xlsx" \\
        --snapshot "Data/M67 westbound between J4 and J3 mainCarriageway 115030402.csv" \\
        --capacity 6649 --length 2713.8037 \\
        --start 2024-09-01 --end 2024-09-30 \\
        --output outputs/finaldata/
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from utils.data import build_finaldata, finaldata_qc_report


def main():
    parser = argparse.ArgumentParser(
        description='构建指定路段的FinalData',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m pipelines.build_finaldata --link 115030402 \\
      --preclean "Data/Precleaned_M67_Traffic_Data_September_2024.xlsx" \\
      --capacity 6649 --length 2713.8037 \\
      --start 2024-09-01 --end 2024-09-30
        """
    )
    
    parser.add_argument('--link', type=int, required=True,
                        help='路段ID (例如: 115030402)')
    parser.add_argument('--preclean', type=str, required=True,
                        help='Precleaned数据文件路径')
    parser.add_argument('--snapshot', type=str, default=None,
                        help='Snapshot CSV文件路径（可选）')
    parser.add_argument('--capacity', type=int, default=None,
                        help='路段容量 (veh/hr)')
    parser.add_argument('--length', type=float, default=None,
                        help='路段长度 (米)')
    parser.add_argument('--start', type=str, default=None,
                        help='起始日期 (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=None,
                        help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--t0-strategy', type=str, default='min5pct',
                        choices=['min5pct', 'low_volume'],
                        help='自由流时间估计策略')
    parser.add_argument('--winsor', type=float, nargs=2, default=[0.01, 0.99],
                        metavar=('LOWER', 'UPPER'),
                        help='Winsorize百分位数 (默认: 0.01 0.99)')
    parser.add_argument('--output', type=str, default='outputs/finaldata/',
                        help='输出目录')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("BPR FinalData构建工具")
    print("="*60)
    print(f"\n参数:")
    print(f"  路段ID: {args.link}")
    print(f"  Precleaned文件: {args.preclean}")
    print(f"  容量: {args.capacity}")
    print(f"  长度: {args.length} m")
    print(f"  时间范围: {args.start} 至 {args.end}")
    print(f"  t0策略: {args.t0_strategy}")
    print(f"  Winsor: {args.winsor}")
    print()
    
    # 构建FinalData
    try:
        df = build_finaldata(
            link_id=args.link,
            precleaned_path=args.preclean,
            snapshot_csv_path=args.snapshot,
            capacity=args.capacity,
            link_length_m=args.length,
            month_start=args.start,
            month_end=args.end,
            t0_strategy=args.t0_strategy,
            winsor=tuple(args.winsor)
        )
        
        # 生成QC报告
        print("\n" + "="*60)
        print("生成QC报告...")
        print("="*60)
        qc_report = finaldata_qc_report(df)
        print(qc_report.to_string(index=False))
        
        # 保存数据
        output_file = output_dir / f"finaldata_{args.link}.parquet"
        df.to_parquet(output_file, index=False)
        print(f"\n✓ FinalData已保存到: {output_file}")
        
        # 保存QC报告
        qc_file = output_dir / f"qc_report_{args.link}.csv"
        qc_report.to_csv(qc_file, index=False)
        print(f"✓ QC报告已保存到: {qc_file}")
        
        # 输出摘要
        print("\n" + "="*60)
        print("构建完成！")
        print("="*60)
        print(f"\n数据摘要:")
        print(f"  总记录数: {len(df)}")
        print(f"  时间范围: {df['datetime'].min()} 至 {df['datetime'].max()}")
        print(f"  平均流量: {df['flow_veh_hr'].mean():.0f} veh/hr")
        print(f"  平均V/C: {df['v_over_c'].mean():.3f}")
        print(f"  平均行程时间: {df['fused_tt_15min'].mean():.2f} 秒")
        print(f"  自由流行程时间: {df['t0_ff'].iloc[0]:.2f} 秒")
        print(f"  有效记录: {df['is_valid'].sum()} ({df['is_valid'].mean()*100:.1f}%)")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

