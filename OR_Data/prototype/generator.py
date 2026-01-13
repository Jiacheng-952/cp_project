import pandas as pd
import numpy as np
import random
from typing import Dict, List, Tuple, Optional
import string


class RailDataGenerator:
    """
    Railway scheduling problem data generator

    Generates CSV files with the following 4 data sheets:
    1. station: Station information (station names and distances)
    2. train: Train information (train numbers, speed classes, stopping patterns)
    3. runtime: Section running times (running times for different speed classes on each section)
    4. parameter: System parameters (time limit, minimum headway, etc.)
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
        Initialize data generator

        Args:
            num_stations: Number of stations (including origin and destination)
            num_trains: Number of trains
            speed_classes: List of speed classes (string format)
            max_distance: Total line length (kilometers)
            time_limit: Time limit (minutes)
            min_headway: Minimum headway time (minutes)
            min_stop_time: Minimum stop time at stations (minutes)
            max_stop_time: Maximum stop time at stations (minutes)
            stop_probability: Probability of stopping at intermediate stations
            random_seed: Random seed for reproducibility
        """
        self.num_stations = max(3, num_stations)  # at least 3 stations
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
        """Generate station names"""
        # Use letters A, B, C... as station names
        if self.num_stations <= 26:
            return [chr(ord("A") + i) for i in range(self.num_stations)]
        else:
            # If more than 26 stations, use S1, S2, S3...
            return [f"S{i+1}" for i in range(self.num_stations)]

    def generate_station_data(self) -> pd.DataFrame:
        """Generate station data"""
        station_names = self.generate_station_names()

        # Generate cumulative distances, ensure monotonic increase
        distances = sorted(np.random.uniform(0, self.max_distance, self.num_stations))
        distances[0] = 0  # origin station distance is 0
        distances[-1] = (
            self.max_distance
        )  # destination station distance is max distance

        return pd.DataFrame(
            {"station": station_names, "mile": [int(d) for d in distances]}
        )

    def generate_train_data(self, station_names: List[str]) -> pd.DataFrame:
        """Generate train data"""
        train_data = []

        for i in range(self.num_trains):
            # Generate train number
            train_no = f"G{i*2+1}"  # G1, G3, G5...

            # Randomly select speed class
            speed = random.choice(self.speed_classes)

            # Generate stopping pattern
            stop_pattern = {"trainNO": train_no, "speed": speed}

            # Origin and destination stations must have stops
            for j, station in enumerate(station_names):
                if j == 0 or j == len(station_names) - 1:
                    stop_pattern[station] = 1  # origin and destination must stop
                else:
                    # Intermediate stations stop with probability
                    stop_pattern[station] = (
                        1 if random.random() < self.stop_probability else 0
                    )

            train_data.append(stop_pattern)

        return pd.DataFrame(train_data)

    def generate_runtime_data(self, station_names: List[str]) -> pd.DataFrame:
        """Generate section runtime data"""
        runtime_data = []

        # Generate runtime for each adjacent station pair
        for i in range(len(station_names) - 1):
            from_station = station_names[i]
            to_station = station_names[i + 1]
            station_pair = f"{from_station}-{to_station}"

            runtime_row = {"station": station_pair}

            # Generate runtime for each speed class
            # Higher speed classes have shorter runtimes
            base_time = random.randint(6, 20)  # base runtime

            for speed_class in self.speed_classes:
                speed_value = int(speed_class)
                # Higher speed means shorter runtime
                time_factor = 300 / speed_value  # use 300 as baseline
                runtime = max(1, int(base_time * time_factor))
                runtime_row[speed_class] = runtime

            runtime_data.append(runtime_row)

        return pd.DataFrame(runtime_data)

    def generate_parameter_data(self) -> pd.DataFrame:
        """Generate parameter data (transposed format)"""
        # Note: parameter.csv is in transposed format
        # First row contains parameter names, second row contains values
        parameters = {
            "T": self.time_limit,
            "H": self.min_headway,
            "MINSTOP": self.min_stop_time,
            "MAXSTOP": self.max_stop_time,
        }

        # Create transposed format DataFrame
        param_names = list(parameters.keys())
        param_values = list(parameters.values())

        return pd.DataFrame([param_names, param_values])

    def generate_data(
        self, output_file: str = "generated_data.xlsx"
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate complete dataset and save to Excel file

        Args:
            output_file: Output file path

        Returns:
            Dictionary containing all worksheet data
        """
        print(f"Generating data...")
        print(f"- Number of stations: {self.num_stations}")
        print(f"- Number of trains: {self.num_trains}")
        print(f"- Speed classes: {self.speed_classes}")
        print(f"- Line length: {self.max_distance}km")

        # Generate data for each worksheet
        station_df = self.generate_station_data()
        station_names = station_df["station"].tolist()

        train_df = self.generate_train_data(station_names)
        runtime_df = self.generate_runtime_data(station_names)
        parameter_df = self.generate_parameter_data()

        # Save to Excel file
        with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
            station_df.to_excel(writer, sheet_name="station", index=False)
            train_df.to_excel(writer, sheet_name="train", index=False)
            runtime_df.to_excel(writer, sheet_name="runtime", index=False)
            parameter_df.to_excel(
                writer, sheet_name="parameter", index=False, header=False
            )

        print(f"Data saved to: {output_file}")

        # Return data dictionary
        return {
            "station": station_df,
            "train": train_df,
            "runtime": runtime_df,
            "parameter": parameter_df,
        }

    def generate_csv_data(self, output_dir: str = "data/") -> Dict[str, pd.DataFrame]:
        """
        Generate complete dataset and save as CSV files to specified directory

        Args:
            output_dir: Output directory path

        Returns:
            Dictionary containing all data
        """
        import os

        print(f"Generating data...")
        print(f"- Number of stations: {self.num_stations}")
        print(f"- Number of trains: {self.num_trains}")
        print(f"- Speed classes: {self.speed_classes}")
        print(f"- Line length: {self.max_distance}km")

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Generate data for each worksheet
        station_df = self.generate_station_data()
        station_names = station_df["station"].tolist()

        train_df = self.generate_train_data(station_names)
        runtime_df = self.generate_runtime_data(station_names)
        parameter_df = self.generate_parameter_data()

        # Save as CSV files
        station_df.to_csv(f"{output_dir}station.csv", index=False)
        train_df.to_csv(f"{output_dir}train.csv", index=False)
        runtime_df.to_csv(f"{output_dir}runtime.csv", index=False)
        parameter_df.to_csv(f"{output_dir}parameter.csv", index=False, header=False)

        print(f"✅ CSV files saved to: {output_dir}")
        print(f"   - {output_dir}station.csv")
        print(f"   - {output_dir}train.csv")
        print(f"   - {output_dir}runtime.csv")
        print(f"   - {output_dir}parameter.csv")

        # Return data dictionary
        return {
            "station": station_df,
            "train": train_df,
            "runtime": runtime_df,
            "parameter": parameter_df,
        }

    def print_data_summary(self, data: Dict[str, pd.DataFrame]):
        """Print data summary"""
        print("\n=== Data Summary ===")

        print(f"\n1. Station Information ({len(data['station'])} stations):")
        print(data["station"].to_string(index=False))

        print(f"\n2. Train Information ({len(data['train'])} trains):")
        print(data["train"].to_string(index=False))

        print(f"\n3. Section Running Times ({len(data['runtime'])} sections):")
        print(data["runtime"].to_string(index=False))

        print(f"\n4. System Parameters:")
        param_df = data["parameter"]
        param_names = param_df.iloc[0].tolist()
        param_values = param_df.iloc[1].tolist()
        for name, value in zip(param_names, param_values):
            print(f"   {name}: {value}")


def create_small_instance():
    """Create small scale instance"""
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
    """Create medium scale instance"""
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
    """Create large scale instance"""
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
    # Example usage
    print("=== Railway Scheduling Problem Data Generator ===\n")

    print("\n" + "=" * 70)
    print(
        "Example: Generate CSV files to data/ directory (can be used directly with solve.py)"
    )
    print("-" * 50)
    generator2 = RailDataGenerator(
        num_stations=10,
        num_trains=10,
        speed_classes=["300", "350"],
        max_distance=200,
        time_limit=200,
        stop_probability=0.4,
        random_seed=456,
    )
    data2 = generator2.generate_csv_data("data/")
    generator2.print_data_summary(data2)

    print("\n" + "=" * 70)
    print("✅ Data generation completed!")
    print("   - CSV format: data/*.csv (can run python solve.py directly)")
    print("=" * 70)
