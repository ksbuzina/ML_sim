"""DQ Report."""

from typing import Dict, List, Tuple, Union
from dataclasses import dataclass
from user_input.metrics import Metric

import pandas as pd

LimitType = Dict[str, Tuple[float, float]]
CheckType = Tuple[str, Metric, LimitType]


class Report:
    """DQ report class."""

    checklist: List[CheckType]
    engine: str = "pandas"

    def __init__(self, checklist, engine="pandas"):
        self.checklist = checklist
        self.engine = engine

    def fit(self, tables: Dict[str, pd.DataFrame]) -> Dict:
        """Calculate DQ metrics and build report."""
        self.report_ = {
            "title": "",
            "result": pd.DataFrame(),
            "passed": 0,
            "passed_pct": 0,
            "failed": 0,
            "failed_pct": 0,
            "errors": 0,
            "errors_pct": 0,
            "total": len(self.checklist)
        }
        report = self.report_
        tables_name = list()
        for table_name, metric, limits in self.checklist:
            try:
                values  = metric(tables[table_name])
                if limits == {}:
                    status = "."
                    error = ''
                else:
                    for key, value in limits.items():
                        min_val, max_val = value
                    if key in values and min_val <= values[key] <= max_val:
                        status = "."
                        error = ''
                    else:
                        status = "F"
                        error = ''
            except Exception as e:
                values = {}
                status = "E"
                error = str(e)
            
            result = {
                "table_name": table_name,
                "metric": repr(metric),
                "limits": str(limits),
                "values": values,
                "status": status,
                "error": str(error),
            }
            if report["result"].empty:
                report["result"] = pd.DataFrame(columns=result.keys())
                report["result"] = report["result"].append(
                    result, ignore_index=True)
            else:
                report["result"] = report["result"].append(
                    result, ignore_index=True)

            if status == ".":
                report["passed"] += 1
            elif status == "F":
                report["failed"] += 1
            elif status == "E":
                report["errors"] += 1
            if table_name not in tables_name:
                tables_name.append(table_name)

        report["passed_pct"] = round((report["passed"]/report["total"])*100, 2)
        report["failed_pct"] = round((report["failed"]/report["total"])*100, 2)
        report["errors_pct"] = round((report["errors"]/report["total"])*100, 2)

        report["title"] = f"DQ Report for tables {sorted(tables.keys())}"
        self.report_ = report
        return report

    def to_str(self) -> None:
        """Convert report to string format."""
        report = self.report_

        msg = (
            "This Report instance is not fitted yet. "
            "Call 'fit' before usong this method."
        )

        assert isinstance(report, dict), msg

        pd.set_option("display.max_rows", 500)
        pd.set_option("display.max_columns", 500)
        pd.set_option("display.max_colwidth", 20)
        pd.set_option("display.width", 1000)

        return (
            f"{report['title']}\n\n"
            f"{report['result']}\n\n"
            f"Passed: {report['passed']} ({report['passed_pct']}%)\n"
            f"Failed: {report['failed']} ({report['failed_pct']}%)\n"
            f"Errors: {report['errors']} ({report['errors_pct']}%)\n"
            "\n"
            f"Total: {report['total']}"
        )
