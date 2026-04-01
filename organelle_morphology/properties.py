from __future__ import annotations
from typing import TYPE_CHECKING, List, Optional
import pandas as pd
import numpy as np
import fnmatch

if TYPE_CHECKING:
    from organelle_morphology.project import Project


class Properties:
    """
    Stores and aggregates morphological properties from a Project in central DataFrames.
    """

    def __init__(self, project: Project):
        self.project = project
        # Instance-level data (Rows: Organelle IDs)
        self._prop_df = pd.DataFrame(index=pd.Index([], name="ID"))
        # Population/Aggregate-level data (Rows: Subset Names/Filters)
        self._agg_df = pd.DataFrame(index=pd.Index([], name="Subset"))

    def update_properties(self, new_data: pd.DataFrame):
        """
        Updates the central instance DataFrame with new properties.
        Expects a DataFrame where the index represents the organelle ID.
        """
        if new_data.index.name != "ID":
            new_data.index.name = "ID"

        if self._prop_df.empty:
            self._prop_df = new_data
        else:
            # combine_first gives priority to the calling dataframe (new_data),
            # updating existing values and appending new columns/rows.
            self._prop_df = new_data.combine_first(self._prop_df)

    def update_aggregates(self, new_data: pd.DataFrame):
        """
        Updates the central aggregate DataFrame with new population metrics.
        Expects a DataFrame where the index represents the subset name/label.
        """
        if new_data.index.name != "Subset":
            new_data.index.name = "Subset"

        if self._agg_df.empty:
            self._agg_df = new_data
        else:
            self._agg_df = new_data.combine_first(self._agg_df)

    def get_available_properties(self) -> list[str]:
        """Returns a list of all available instance properties currently in the dataframe."""
        return self._prop_df.columns.tolist()

    def get_aggregate_properties(self) -> list[str]:
        """Returns a list of all available aggregate properties currently in the dataframe."""
        return self._agg_df.columns.tolist()

    def get_properties(
        self, ids: str = "*", properties: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Returns a unified DataFrame with a user-defined selection of properties.
        check get_properties to see all currently available properties.

        :param ids: Glob-style filter for organelles (e.g., "mito_*").
        :param properties: List of keys to include. If None, returns all available.
        """
        if self._prop_df.empty:
            return pd.DataFrame(columns=["ID"] + (properties if properties else []))

        # Filter by IDs using glob matching
        if ids == "*":
            filtered_df = self._prop_df
        else:
            matched_ids = [
                idx for idx in self._prop_df.index if fnmatch.fnmatch(idx, ids)
            ]
            filtered_df = self._prop_df.loc[matched_ids]

        # Filter by requested properties
        if properties is not None:
            selected_set = set(properties)
            final_cols = []

            for col in filtered_df.columns:
                # Check for exact match (e.g., "mesh_volume")
                if col in selected_set:
                    final_cols.append(col)
                    continue

                # Check for dynamic contact columns (e.g., "0-0.01-n_contacts")
                # where the generic base property ("n_contacts") was requested
                parts = col.rsplit("-", 1)
                if len(parts) == 2 and parts[1] in selected_set:
                    final_cols.append(col)

            filtered_df = filtered_df[final_cols]

        # Reset index to make 'ID' a standard column to match the old behavior
        return filtered_df.reset_index()

    def _calculate_stats(self, df: pd.DataFrame, selection: str = "*") -> dict:
        """
        Pure calculation method. Returns a dictionary of statistical series.
        """
        if "ID" in df.columns:
            calc_df = df.drop(columns=["ID"])
        else:
            calc_df = df.copy()

        numeric_cols = calc_df.select_dtypes(include=[np.number]).columns.tolist()
        bool_cols = calc_df.select_dtypes(include=[bool]).columns.tolist()

        if not (numeric_cols or bool_cols):
            return {}

        for col in bool_cols:
            calc_df[col] = calc_df[col].astype(float)

        stats_rows = {}
        stats_rows["Count"] = calc_df[numeric_cols + bool_cols].count()
        stats_rows["Sum"] = calc_df[numeric_cols + bool_cols].sum()
        stats_rows["Average"] = calc_df[numeric_cols + bool_cols].mean()

        numeric_df = calc_df[numeric_cols]
        if not numeric_df.empty:
            stats_rows["Fraction_Non_Zero"] = (numeric_df > 0).mean()
            stats_rows["Std_Dev"] = numeric_df.std()
            stats_rows["Median"] = numeric_df.median()
            stats_rows["Minimum"] = numeric_df.min()
            stats_rows["Maximum"] = numeric_df.max()
            stats_rows["16th_percentile"] = numeric_df.quantile(0.16)
            stats_rows["84th_percentile"] = numeric_df.quantile(0.84)
            stats_rows["Geometric_Mean"] = numeric_df.apply(
                lambda x: np.exp(np.log(x[x > 0]).mean()) if np.any(x > 0) else np.nan
            )

        # Flatten and permanently store in the central aggregate DataFrame
        flat_stats = {}
        for measure_name, series in stats_rows.items():
            for col_name, value in series.items():
                flat_stats[f"{col_name}_{measure_name}"] = value

        agg_update_df = pd.DataFrame(
            [flat_stats], index=pd.Index([selection], name="Subset")
        )
        self.update_aggregates(agg_update_df)

        return stats_rows

    def get_summary_dataframe(
        self, ids: str = "*", properties: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Calculates statistics for a selection of organelles and properties,
        triggers the aggregate storage, and returns the UI-friendly 2D summary dataframe.
        """
        # Fetch the filtered dataframe internally
        df = self.get_dataframe(ids=ids, properties=properties)

        # Calculate stats, store in _agg_df using 'ids' as the subset name,
        # and get the raw series back
        stats_rows = self.calculate_stats(df, selection=ids)
        if not stats_rows:
            return pd.DataFrame()

        summary_df = pd.DataFrame(stats_rows).T

        # Rename the boolean Average back to "Average (or Share)" for the UI display
        summary_df = summary_df.rename(index={"Average": "Average (or Share)"})

        # Ensure the columns match the order of the input dataframe (ignoring ID)
        calc_df_cols = [c for c in df.columns if c != "ID"]
        target_order = [c for c in calc_df_cols if c in summary_df.columns]

        summary_df = summary_df.reindex(columns=target_order)
        summary_df.index.name = "Measure"
        summary_df = summary_df.reset_index()

        return summary_df
