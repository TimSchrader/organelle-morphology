import pandas as pd


def test_statistics_defaults(project_with_sources):
    """Verify that we get a basic dataframe with default properties."""
    project_with_sources.compute_geometry()
    stats = project_with_sources.properties
    df = stats.get_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "ID" in df.columns
    # Default properties check
    assert "mesh_volume" in df.columns
    assert "sphericity" in df.columns
    assert df.columns[0] == "ID"


def test_statistics_alphabetical_sorting(project_with_sources):
    """Check that columns are sorted and ID filtering works."""
    project_with_sources.compute_geometry()
    stats = project_with_sources.properties
    props = ["water_tight", "mesh_area", "mesh_volume"]  # non-alphabetical order
    df = stats.get_dataframe(ids="mito_0001", properties=props)
    assert df.shape[0] == 1
    assert df.iloc[0]["ID"] == "mito_0001"
    expected_order = [
        "ID",
        "mesh_area",
        "mesh_volume",
        "water_tight",
    ]  # alphabetical order
    assert list(df.columns) == expected_order


def test_statistics_summary_dataframe(project_with_sources):
    """Test the statistical summary logic, including boolean handling."""
    project_with_sources.compute_geometry()
    stats = project_with_sources.properties

    # water_tight is boolean
    summary_df = stats.get_summary_dataframe(
        properties=["mesh_volume", "water_tight", "sphericity"]
    )

    assert "Measure" in summary_df.columns
    assert "Average (or Share)" in summary_df["Measure"].values
    avg_row = summary_df[summary_df["Measure"] == "Average (or Share)"].iloc[0]
    assert isinstance(avg_row["mesh_volume"], float)  # Volume should be a float
    assert 0.0 <= avg_row["water_tight"] <= 1.0  # Share of water_tight
    std_row = summary_df[summary_df["Measure"] == "Std_Dev"].iloc[0]
    assert pd.isna(
        std_row["water_tight"]
    )  # Standard deviation should be NaN for boolean columns


def test_statistics_mcs_aggregation(project_with_sources):
    """Verify that MCS data is correctly pulled from the central dataframe."""
    project_with_sources.compute_geometry()
    stats = project_with_sources.properties

    # Instead of mocking organelle dicts, we directly inject data to the central df
    test_mcs_data = pd.DataFrame(
        [{"ID": "mito_0001", "0-0.01-total_area": 150.5, "0-0.01-mean_dist": 42.0}]
    ).set_index("ID")
    stats.update(test_mcs_data)

    props = ["0-0.01-total_area", "0-0.01-mean_dist", "mesh_volume"]
    df = stats.get_dataframe(ids="mito_0001", properties=props)
    assert df.iloc[0]["0-0.01-total_area"] == 150.5  # Verify
    assert df.iloc[0]["0-0.01-mean_dist"] == 42.0  # Verify

    summary_df = stats.get_summary_dataframe(ids="mito_0001", properties=props)
    avg_row = summary_df[summary_df["Measure"] == "Average (or Share)"].iloc[0]
    assert avg_row["0-0.01-total_area"] == 150.5


def test_statistics_integration(project_with_sources):
    """Integration test for the full statistics pipeline."""
    # Compute all data streams
    project_with_sources.compute_geometry()
    project_with_sources.skeletonize_wavefront()
    project_with_sources.search_mcs(10)  # calc and add contact sites

    stats = project_with_sources.properties
    stats_df = stats.get_dataframe()  # Get all calculated properties

    assert not stats_df.empty
    cols = stats_df.columns.tolist()

    # Check that static properties arrived
    assert "mesh_volume" in cols, "Static property mesh_volume is missing."
    assert "num_nodes" in cols, "Skeleton property num_nodes is missing."
    assert stats_df["mesh_volume"].notna().all(), "Some data for mesh_volume is NaN."

    # Check that dynamic MCS properties arrived
    mcs_cols = [c for c in cols if "n_contacts" in c]
    assert len(mcs_cols) > 0, "MCS columns were not generated."
    assert stats_df[mcs_cols[0]].notna().any(), "MCS data is entirely NaN."
