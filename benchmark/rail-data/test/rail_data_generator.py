import pandas as pd
import numpy as np
import random
from typing import Dict, List, Tuple, Optional
import string


class RailDataGenerator:
    """
    高铁排班问题数据生成器

    生成包含以下4个工作表的Excel文件：
    1. station: 车站信息（车站名称和里程）
    2. train: 列车信息（车次、速度等级、停站标识）
    3. runtime: 区间运行时间（不同速度等级在各区间的运行时间）
    4. parameter: 系统参数（时间上限、最小间隔等）
    """

    def __init__(
        self,
        num_stations: int = 7,
        num_trains: int = 5,
        speed_classes: List[str] = ["300", "350"],
        max_distance: int = 300,
        time_limit: int = 160,
        min_headway: int = 5,
        min_stop_time: int = 2,
        max_stop_time: int = 15,
        stop_probability: float = 0.3,
        random_seed: Optional[int] = None,
    ):
        """
        初始化数据生成器

        Args:
            num_stations: 车站数量（包括起点和终点）
            num_trains: 列车数量
            speed_classes: 速度等级列表（字符串格式）
            max_distance: 线路总长度（公里）
            time_limit: 时间上限（分钟）
            min_headway: 最小间隔时间（分钟）
            min_stop_time: 最小停站时间（分钟）
            max_stop_time: 最大停站时间（分钟）
            stop_probability: 中间站停站概率
            random_seed: 随机种子
        """
        self.num_stations = max(3, num_stations)  # 至少3个站
        self.num_trains = max(1, num_trains)
        self.speed_classes = speed_classes
        self.max_distance = max_distance
        self.time_limit = time_limit
        self.min_headway = min_headway
        self.min_stop_time = min_stop_time
        self.max_stop_time = max_stop_time
        self.stop_probability = stop_probability

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

    def generate_station_names(self) -> List[str]:
        """生成车站名称"""
        # 使用字母A, B, C...作为车站名称
        if self.num_stations <= 26:
            return [chr(ord("A") + i) for i in range(self.num_stations)]
        else:
            # 如果超过26个站，使用S1, S2, S3...
            return [f"S{i+1}" for i in range(self.num_stations)]

    def generate_station_data(self) -> pd.DataFrame:
        """生成车站工作表数据"""
        station_names = self.generate_station_names()

        # 生成累积里程，确保单调递增
        distances = sorted(np.random.uniform(0, self.max_distance, self.num_stations))
        distances[0] = 0  # 起点里程为0
        distances[-1] = self.max_distance  # 终点里程为最大距离

        return pd.DataFrame(
            {"station": station_names, "mile": [int(d) for d in distances]}
        )

    def generate_train_data(self, station_names: List[str]) -> pd.DataFrame:
        """生成列车工作表数据"""
        train_data = []

        for i in range(self.num_trains):
            # 生成车次号
            train_no = f"G{i*2+1}"  # G1, G3, G5...

            # 随机选择速度等级
            speed = random.choice(self.speed_classes)

            # 生成停站标识
            stop_pattern = {"trainNO": train_no, "speed": speed}

            # 起点和终点必须停站
            for j, station in enumerate(station_names):
                if j == 0 or j == len(station_names) - 1:
                    stop_pattern[station] = 1  # 起点和终点必停
                else:
                    # 中间站按概率停站
                    stop_pattern[station] = (
                        1 if random.random() < self.stop_probability else 0
                    )

            train_data.append(stop_pattern)

        return pd.DataFrame(train_data)

    def generate_runtime_data(self, station_names: List[str]) -> pd.DataFrame:
        """生成区间运行时间工作表数据"""
        runtime_data = []

        # 为每个相邻车站对生成运行时间
        for i in range(len(station_names) - 1):
            from_station = station_names[i]
            to_station = station_names[i + 1]
            station_pair = f"{from_station}-{to_station}"

            runtime_row = {"station": station_pair}

            # 为每个速度等级生成运行时间
            # 高速度等级的运行时间更短
            base_time = random.randint(6, 20)  # 基础运行时间

            for speed_class in self.speed_classes:
                speed_value = int(speed_class)
                # 速度越高，运行时间越短
                time_factor = 300 / speed_value  # 以300为基准
                runtime = max(1, int(base_time * time_factor))
                runtime_row[speed_class] = runtime

            runtime_data.append(runtime_row)

        return pd.DataFrame(runtime_data)

    def generate_parameter_data(self) -> pd.DataFrame:
        """生成参数工作表数据（转置格式）"""
        # 注意：根据test.py，parameter工作表是转置格式
        # 第一行是参数名，第二行是参数值
        parameters = {
            "T": self.time_limit,
            "H": self.min_headway,
            "MINSTOP": self.min_stop_time,
            "MAXSTOP": self.max_stop_time,
        }

        # 创建转置格式的DataFrame
        param_names = list(parameters.keys())
        param_values = list(parameters.values())

        return pd.DataFrame([param_names, param_values])

    def generate_data(
        self, output_file: str = "generated_data.xlsx"
    ) -> Dict[str, pd.DataFrame]:
        """
        生成完整的数据集并保存到Excel文件

        Args:
            output_file: 输出文件路径

        Returns:
            包含所有工作表数据的字典
        """
        print(f"正在生成数据...")
        print(f"- 车站数量: {self.num_stations}")
        print(f"- 列车数量: {self.num_trains}")
        print(f"- 速度等级: {self.speed_classes}")
        print(f"- 线路长度: {self.max_distance}km")

        # 生成各个工作表的数据
        station_df = self.generate_station_data()
        station_names = station_df["station"].tolist()

        train_df = self.generate_train_data(station_names)
        runtime_df = self.generate_runtime_data(station_names)
        parameter_df = self.generate_parameter_data()

        # 保存到Excel文件
        with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
            station_df.to_excel(writer, sheet_name="station", index=False)
            train_df.to_excel(writer, sheet_name="train", index=False)
            runtime_df.to_excel(writer, sheet_name="runtime", index=False)
            parameter_df.to_excel(
                writer, sheet_name="parameter", index=False, header=False
            )

        print(f"数据已保存到: {output_file}")

        # 返回数据字典
        return {
            "station": station_df,
            "train": train_df,
            "runtime": runtime_df,
            "parameter": parameter_df,
        }

    def print_data_summary(self, data: Dict[str, pd.DataFrame]):
        """打印数据摘要"""
        print("\n=== 数据摘要 ===")

        print(f"\n1. 车站信息 ({len(data['station'])} 个车站):")
        print(data["station"].to_string(index=False))

        print(f"\n2. 列车信息 ({len(data['train'])} 列列车):")
        print(data["train"].to_string(index=False))

        print(f"\n3. 区间运行时间 ({len(data['runtime'])} 个区间):")
        print(data["runtime"].to_string(index=False))

        print(f"\n4. 系统参数:")
        param_df = data["parameter"]
        param_names = param_df.iloc[0].tolist()
        param_values = param_df.iloc[1].tolist()
        for name, value in zip(param_names, param_values):
            print(f"   {name}: {value}")


def create_small_instance():
    """创建小规模实例"""
    generator = RailDataGenerator(
        num_stations=5,
        num_trains=3,
        speed_classes=["300", "350"],
        max_distance=200,
        time_limit=120,
        stop_probability=0.4,
        random_seed=42,
    )
    return generator.generate_data("small_instance.xlsx")


def create_medium_instance():
    """创建中等规模实例"""
    generator = RailDataGenerator(
        num_stations=8,
        num_trains=6,
        speed_classes=["250", "300", "350"],
        max_distance=400,
        time_limit=200,
        stop_probability=0.3,
        random_seed=42,
    )
    return generator.generate_data("medium_instance.xlsx")


def create_large_instance():
    """创建大规模实例"""
    generator = RailDataGenerator(
        num_stations=12,
        num_trains=10,
        speed_classes=["200", "250", "300", "350"],
        max_distance=600,
        time_limit=300,
        stop_probability=0.25,
        random_seed=42,
    )
    return generator.generate_data("large_instance.xlsx")


if __name__ == "__main__":
    # 示例用法
    print("=== 高铁排班问题数据生成器 ===\n")

    # 创建不同规模的实例
    print("1. 创建小规模实例...")
    small_data = create_small_instance()

    print("\n" + "=" * 50 + "\n")

    print("2. 创建中等规模实例...")
    medium_data = create_medium_instance()

    print("\n" + "=" * 50 + "\n")

    print("3. 创建大规模实例...")
    large_data = create_large_instance()

    # 展示小规模实例的详细信息
    print("\n" + "=" * 50)
    print("小规模实例详细信息:")
    generator = RailDataGenerator(num_stations=5, num_trains=3, random_seed=42)
    generator.print_data_summary(small_data)
