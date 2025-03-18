import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.joinpath("..")))
from utils.input_utils import get_file_paths


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="load an decode the json_predictions to arrays")
    io_args = parser.add_argument_group("IO")
    io_args.add_argument("-i", "--input", nargs="+", action="extend", help="Input file(s)", required=True, type=str)

    extraction_args = parser.add_argument_group("Extraction")
    extraction_args.add_argument("-m", "--metric", help="Metric to be extracted", required=True, type=str)
    extraction_args.add_argument("-r", "--return_type", help="Return value", type=str, default="best", choices=["all", "best"])
    extraction_args.add_argument("--sort_method", help="Sort method", type=str, default="max", choices=["max", "min"])
    extraction_args.add_argument("--print_all", help="Print all values", action="store_true")

    args = parser.parse_args()
    return args


def main(args):
    json_paths = get_file_paths(args.input, formats=[".json"])

    for json_path in json_paths:
        with json_path.open(mode="r") as f:
            things_to_print = []
            best_value = None
            best_metrics = None
            for line in f:
                metrics = json.loads(line)

                if args.metric in metrics:
                    if args.return_type == "all":
                        if args.print_all:
                            things_to_print.append(metrics)
                        else:
                            things_to_print.append({args.metric: metrics[args.metric]})
                    elif args.return_type == "best":
                        if args.sort_method == "max":
                            if best_value is None or metrics[args.metric] > best_value:
                                best_value = metrics[args.metric]
                                best_metrics = metrics
                        elif args.sort_method == "min":
                            if best_value is None or metrics[args.metric] < best_value:
                                best_value = metrics[args.metric]
                                best_metrics = metrics
                        else:
                            raise ValueError(f"Invalid sort method: {args.sort_method}")
                    else:
                        raise ValueError(f"Invalid return type: {args.return_type}")
            if args.print_all:
                things_to_print.append(best_metrics)
            else:
                things_to_print.append({args.metric: best_value})

        print(f"File: {json_path} - Metric: {args.metric} - Return type: {args.return_type} - Sort method: {args.sort_method}")
        for thing in things_to_print:
            print(thing)
        print()


if __name__ == "__main__":
    args = get_arguments()
    main(args)
