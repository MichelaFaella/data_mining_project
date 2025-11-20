import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
from IPython.display import display


def plot_birth_decades(df, title1, title2):
    """
    Plots:
      1. Histogram of birth years
      2. Percentage of unique artists by decade of birth

    Parameters:
        df (pd.DataFrame): DataFrame containing at least 'name' and 'birth_date' columns.
    """
    # --- 1. Keep only unique artists ---
    unique_artists = df.drop_duplicates(subset=["name"])

    # --- 2. Extract birth year ---
    birth_year = unique_artists["birth_date"].dt.year

    # Drop missing values
    birth_year = birth_year.dropna()

    # --- 4. Create decade bins ---
    start = (int(birth_year.min()) // 10) * 10
    end = (int(birth_year.max()) // 10 + 1) * 10
    bins = list(range(start, end + 1, 10))
    labels = [f"{b}s" for b in bins[:-1]]

    # --- 5. Assign decades ---
    decades = pd.cut(birth_year, bins=bins, labels=labels, right=False)

    # --- 6. Calculate percentage per decade ---
    group_percent = decades.value_counts(normalize=True).sort_index() * 100
    group_df = pd.DataFrame(
        {"decade": group_percent.index, "percent": group_percent.values}
    )

    # --- 7. Plot both charts side by side ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Histogram of birth years
    sns.histplot(birth_year, bins=20, kde=True, ax=axes[0], color="skyblue")
    axes[0].set_title(title1, fontsize=16, pad=15)
    axes[0].set_xlabel("Birth Year", fontsize=12)
    axes[0].set_ylabel("Count", fontsize=12)

    # Percentage per decade
    sns.barplot(
        data=group_df,
        x="decade",
        y="percent",
        hue="decade",
        palette="coolwarm",
        legend=False,
        ax=axes[1],
    )
    for i, val in enumerate(group_df["percent"]):
        axes[1].text(i, val + 0.5, f"{val:.1f}%", ha="center", fontsize=10)
    axes[1].set_title(title2, fontsize=16, pad=15)
    axes[1].set_xlabel("Decade", fontsize=12)
    axes[1].set_ylabel("Percentage (%)", fontsize=12)

    sns.despine()
    plt.tight_layout()
    plt.show()


def plot_artist_ages(df, title):
    """
    Calculates and plots the number of unique artists by age.

    Parameters:
        df (pd.DataFrame): DataFrame containing at least 'name' and 'birth_date' columns.
    """
    # --- 1. Keep only unique artists ---
    unique_artists = df.drop_duplicates(subset=["name"])

    # --- 2. Calculate current age ---
    today = pd.Timestamp.today()
    artist_age = today.year - unique_artists["birth_date"].dt.year

    # --- 3. Drop missing or invalid ages ---
    artist_age = artist_age.dropna().astype(int)

    # --- 4. Count how many unique artists have each exact age ---
    age_counts = artist_age.value_counts().sort_index()
    print("Number of unique artists per age:")
    print(age_counts)

    # --- 5. Plot the distribution ---
    plt.figure(figsize=(8, 5))
    sns.barplot(x=age_counts.index, y=age_counts.values, color="#6A5ACD")

    plt.title(title, fontsize=16, color="#333")
    plt.xlabel("Age (years)", fontsize=12)
    plt.ylabel("Number of Artists", fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_active_start_decades(df, title, active_col="active_start", name_col="name"):
    """
    Plot percentage of unique artists by active start decade.
    """

    # --- Keep only unique artists based on name ---
    unique_df = df.drop_duplicates(subset=[name_col])

    # --- Extract year values, drop missing ---
    years = unique_df[active_col].dropna().dt.year

    # --- Define decade bins (e.g., 1960, 1970, ..., 2020) ---
    start = int(years.min() // 10 * 10)
    end = int(years.max() // 10 * 10 + 10)
    bins = list(range(start, end + 10, 10))
    labels = [f"{b}s" for b in bins[:-1]]

    # --- Bin into decades ---
    decade_groups = pd.cut(years, bins=bins, labels=labels, right=False)

    # --- Calculate percentages per decade ---
    group_percent = decade_groups.value_counts(normalize=True).sort_index() * 100
    group_df = pd.DataFrame(
        {"decade": group_percent.index, "percent": group_percent.values}
    )

    # --- Print results ---
    print(title)
    print(group_df)

    # --- Plot ---
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=group_df,
        x="decade",
        y="percent",
        hue="decade",
        palette="coolwarm",
        legend=False,
    )

    # --- Add percentage labels ---
    for i, val in enumerate(group_df["percent"]):
        plt.text(i, val + 0.5, f"{val:.1f}%", ha="center", fontsize=10)

    plt.title(title, fontsize=18, pad=15)
    plt.xlabel("Decade", fontsize=12)
    plt.ylabel("Percentage (%)", fontsize=12)
    sns.despine()
    plt.tight_layout()
    plt.show()


def plot_age_at_career_start(
    df, title, birth_col="birth_date", active_col="active_start", name_col="name"
):
    """
    Plot the distribution of unique artists' ages when they started their career.
    Does not modify the dataset or filter any values.
    """

    # --- Keep only unique artists based on name ---
    unique_df = df.drop_duplicates(subset=[name_col])

    # --- Compute age at career start ---
    age_at_start = (
        (unique_df[active_col].dt.year - unique_df[birth_col].dt.year)
        .dropna()
        .astype(int)
    )

    # --- Count occurrences ---
    age_counts = age_at_start.value_counts().sort_index()
    print(title)
    print(age_counts)

    # --- Plot ---
    plt.figure(figsize=(10, 8))
    bars = plt.bar(age_counts.index, age_counts.values, color="#6A5ACD")

    # --- Add counts on top of each bar ---
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.5,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.title(title, fontsize=16, color="#333")
    plt.xlabel("Age at Career Start (years)", fontsize=12)
    plt.ylabel("Number of Artists", fontsize=12)
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.show()


def display_artist_ages(df, title="Unique Artists and Their Age"):
    """
    Display a table of unique artists and their ages.

    Parameters:
        df (pd.DataFrame): DataFrame containing at least 'name' and 'birth_date' columns.
        title (str): Custom title for the output.
    """
    # --- Compute current age in years ---
    today = pd.Timestamp.today()
    artist_age = (today - df["birth_date"]).dt.days // 365

    # --- Create temporary DataFrame with unique artists and their age ---
    artist_age_df = (
        pd.DataFrame({"name": df["name"], "age": artist_age})
        .drop_duplicates()
        .sort_values(by="age", ascending=False)
    )

    # --- Display the table ---
    print(title + ":")
    display(artist_age_df.style.background_gradient(cmap="coolwarm"))

    # --- Total number of unique artists ---
    print(f"\nTotal number of unique artists: {artist_age_df['name'].nunique()}")


def safe_int(x):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return None
        if isinstance(x, str) and x.strip() == "":
            return None
        return int(x)
    except Exception:
        return None


TOKEN_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)


def count_tokens(lyrics: str) -> int:

    if not isinstance(lyrics, str):
        return 0
    tokens = TOKEN_PATTERN.findall(lyrics)
    return len(tokens)


def count_sentences(lyrics: str) -> int:
    if not isinstance(lyrics, str):
        return 0
    # Basic heuristic: ., !, ?, ;, : and line breaks as sentence boundaries
    text = lyrics.replace("\r", "\n")
    # Split on punctuation that typically ends a sentence or on newlines
    parts = re.split(r"[\.!\?\n;:]+", text)
    # Count non-empty segments that contain at least one word character
    sent_count = sum(1 for p in parts if re.search(r"\w", p))
    return sent_count


def count_total_token_chars(lyrics: str) -> int:
    """
    Counts the total number of characters that make up valid tokens,
    ignoring spaces and punctuation.
    """
    if not isinstance(lyrics, str):
        return 0
    tokens = TOKEN_PATTERN.findall(lyrics)
    # Somma la lunghezza di ogni token trovato
    total_chars = sum(len(t) for t in tokens)
    return total_chars


def count_unique_tokens(lyrics: str) -> int:
    """
    Counts the number of unique tokens (words) found by our TOKEN_PATTERN.
    Non usiamo .lower() per essere coerenti con come abbiamo calcolato n_tokens_auto.
    """
    if not isinstance(lyrics, str):
        return 0
    tokens = TOKEN_PATTERN.findall(lyrics)
    if not tokens:
        return 0

    return len(set(tokens))


def is_binary_series(s: pd.Series) -> bool:
    """Return True if the series is effectively binary (0/1) ignoring NaNs."""
    vals = s.dropna().unique()
    if len(vals) == 0:
        return False
    try:
        return set(pd.Series(vals).astype(float)).issubset({0.0, 1.0})
    except Exception:
        return False


def is_low_cardinal_int(s: pd.Series, max_levels: int = 5) -> bool:
    """Return True for low-cardinality integer columns (ordinal-like), better for Spearman/Kendall."""
    if not pd.api.types.is_integer_dtype(s):
        return False
    return s.nunique(dropna=True) <= max_levels


def is_quasi_constant(s: pd.Series, tol_unique_ratio: float = 0.001) -> bool:
    """Return True if the column is constant or nearly constant (variance ~ 0)."""
    if s.nunique(dropna=True) <= 1:
        return True
    unique_ratio = s.nunique(dropna=True) / max(1, s.notna().sum())
    return unique_ratio <= tol_unique_ratio


def is_extreme_zeroinflated_count(
    s: pd.Series, zero_ratio_threshold: float = 0.95
) -> bool:
    """Return True for non-negative integer counts with an extreme fraction of zeros."""
    if not (
        pd.api.types.is_integer_dtype(s) or pd.api.types.is_unsigned_integer_dtype(s)
    ):
        return False
    if (s < 0).any():
        return False
    zero_ratio = (s == 0).mean()
    return zero_ratio >= zero_ratio_threshold


def is_quasi_constant(s: pd.Series, tol_unique_ratio: float = 0.001) -> bool:
    """Return True if the column is constant or nearly constant (variance ~ 0)."""
    if s.dropna().nunique() <= 1:
        return True
    unique_ratio = s.dropna().nunique() / max(1, s.notna().sum())
    return unique_ratio <= tol_unique_ratio


def plot_unique_album_release_distribution(
    df,
    title1,
    title2,
    album_col="album",
    date_col="album_release_date",
):
    """
    Plots the percentage of unique albums by their release decade (bar plot)
    and a histogram of unique album release years.
    Does not modify the original dataset.
    """
    # --- Keep only unique albums ---
    unique_albums = df.drop_duplicates(subset=[album_col]).copy()

    # --- Extract album release years ---
    album_years = unique_albums[date_col].dt.year.dropna()

    if album_years.empty:
        print(f"No valid album release years found in column '{date_col}'.")
        return

    # --- Define decade bins ---
    start = int(album_years.min() // 10 * 10)
    end = int(album_years.max() // 10 * 10 + 10)
    bins = list(range(start, end + 10, 10))
    labels = [f"{b}s" for b in bins[:-1]]

    # --- Group by decade ---
    decade_groups = pd.cut(album_years, bins=bins, labels=labels, right=False)

    # --- Calculate percentages ---
    group_percent = decade_groups.value_counts(normalize=True).sort_index() * 100
    group_df = pd.DataFrame(
        {"decade": group_percent.index, "percent": group_percent.values}
    )

    print(title1)
    print(group_df)

    # --- Plot bar chart and histogram side by side ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Bar plot ---
    sns.barplot(
        data=group_df,
        x="decade",
        y="percent",
        hue="decade",
        palette="mako",
        legend=False,
        ax=axes[0],
    )
    for i, val in enumerate(group_df["percent"]):
        axes[0].text(i, val + 0.5, f"{val:.2f}%", ha="center", fontsize=10)

    axes[0].set_title(title1, fontsize=16, pad=15)
    axes[0].set_xlabel("Album Release Decade", fontsize=12)
    axes[0].set_ylabel("Percentage (%)", fontsize=12)
    sns.despine(ax=axes[0])

    # --- Histogram of release years ---
    sns.histplot(album_years, bins=20, kde=True, color="skyblue", ax=axes[1])
    axes[1].set_title(title2, fontsize=16, pad=15)
    axes[1].set_xlabel("Release Year", fontsize=12)
    axes[1].set_ylabel("Count", fontsize=12)
    sns.despine(ax=axes[1])

    plt.tight_layout()
    plt.show()


def find_albums_missing_release_date(
    df, album_col="album", artist_col="name", date_col="album_release_date"
):
    """
    Finds and lists  albums that have a name but no release date.
    """
    missing_tracks = df[df[album_col].notna() & df[date_col].isna()][
        [album_col, artist_col, "album_type", "title", "year"]
    ]

    unique_missing_albums = missing_tracks.drop_duplicates(subset=[album_col])

    print(
        f"Number of unique albums with a name but without release date: {len(unique_missing_albums)}"
    )

    return missing_tracks


def find_albums_before_birth(df):
    """
    Find albums released before the artist's birth date.
    """
    album_before_birth = df[df["album_release_date"] < df["birth_date"]]

    print(f"Albums released before artist's birth: {len(album_before_birth)}")
    return album_before_birth[
        [
            "title",
            "album",
            "album_release_date",
            "birth_date",
            "name",
            "disc_number",
            "track_number",
        ]
    ]


def find_albums_before_active_start(df):
    """
    Find albums released before the artist's active date.
    """
    album_before_active_start = df[df["album_release_date"] < df["active_start"]]

    unique_albums_before_start = album_before_active_start.drop_duplicates(
        subset=["album"]
    )

    print(
        f"Albums released before artist's active start date: {len(album_before_active_start)}"
    )

    print(
        f"unique Albums released before artist's active start date: {len(unique_albums_before_start)}"
    )

    return unique_albums_before_start[
        ["album", "album_release_date", "active_start", "birth_date", "name"]
    ]


def analyze_nan_albums(df):
    """
    Analyze rows where 'album' is NaN:
    - Count how many rows have album = NaN
    - Show breakdown by album_type
    - Count how many are single vs album
    - Display titles/years for the subset where album_type == 'album'

    Returns:
        nan_album_rows (DataFrame)
        nan_and_album_type_album (DataFrame)
    """

    # --- 1. Filter rows where album is NaN ---
    nan_album_rows = df[df["album"].isna()]

    # --- 2. Print total count ---
    print(f"Total number of rows with 'album' = NaN: {len(nan_album_rows)}")

    # --- 3. Breakdown by album_type ---
    print("\nBreakdown by 'album_type':")
    type_counts = nan_album_rows["album_type"].value_counts(dropna=False)
    display(type_counts)

    # Helper counts
    count_single = type_counts.get("single", 0)
    count_album = type_counts.get("album", 0)

    print(f"\nRows with album=NaN and album_type='single': {count_single}")
    print(f"Rows with album=NaN and album_type='album': {count_album}")

    # --- 4. Subset where album_type = 'album' ---
    nan_and_album_type_album = nan_album_rows[nan_album_rows["album_type"] == "album"]

    # Return the two useful DataFrames
    return nan_album_rows, nan_and_album_type_album


def analyze_extreme_years(df):
    """
    Analyze songs with extreme or invalid year values:
    - Songs before 1950
    - Songs after 2025


    Returns:
        old_songs (DataFrame): rows where year < 1950
        future_songs (DataFrame): rows where year > 2025
    """

    # --- Boolean masks for year filtering ---
    before_1950_mask = df["year"] < 1950
    after_2025_mask = df["year"] > 2025

    # Count the number of titles in each category
    songs_before_1950 = df[before_1950_mask]["title"].count()
    songs_after_2025 = df[after_2025_mask]["title"].count()

    # --- Print total counts ---
    print(f"Total number of titles before 1950: {songs_before_1950}")
    print(f"Total number of titles after 2025: {songs_after_2025}")

    # Columns to display
    cols_to_display = [
        "title",
        "album",
        "album_release_date",
        "album_type",
        "year",
        "month",
        "day",
    ]

    # --- Show songs before 1950 ---

    old_songs = df[before_1950_mask]
    # --- Show songs after 2025 ---
    future_songs = df[after_2025_mask]

    return old_songs, future_songs


def find_common_nan_dates(df):
    """
    Find rows where BOTH 'album_release_date' and 'year' are NaN.

    Returns:
        common_nan_rows (DataFrame): rows with both fields missing.
    """

    # --- 1. Mask for album_release_date == NaN ---
    date_nan_mask = df["album_release_date"].isna()

    # --- 2. Mask for year == NaN ---
    year_nan_mask = df["year"].isna()

    # --- 3. Combine masks (both NaN in the same row) ---
    common_nan_mask = date_nan_mask & year_nan_mask

    # --- 4. Filter the DataFrame ---
    common_nan_rows = df[common_nan_mask]

    # --- 5. Reporting ---
    print("--- Common NaNs Check ('album_release_date' AND 'year') ---")
    print(
        f"Number of rows where BOTH 'album_release_date' AND 'year' are NaN: {len(common_nan_rows)}"
    )

    if len(common_nan_rows) > 0:
        print("\nDisplaying up to 20 example rows:")
        display(
            common_nan_rows[["title", "name", "album", "album_release_date", "year"]]
        )
    else:
        print("\nNo rows found with common NaNs in both columns.")

    return common_nan_rows


def find_active_start_before_birthdate(df):
    """
    Find artists whose active_start date is earlier than their birth date.
    """

    invalid_dates = df[df["active_start"] < df["birth_date"]]

    print(
        f"Found {len(invalid_dates)} artists with 'active_start' earlier than 'birth_date'."
    )
    display(invalid_dates[["name", "birth_date", "active_start"]])


def find_tracks_before_career_start(df):
    """
    Find tracks released before the artist's career start.
    """
    inconsistency = df[
        (df["year"].notna())
        & (df["active_start"].notna())
        & (df["year"] < df["active_start"].dt.year)
    ]

    print(
        f"Number of records where a song was released before the artist's career start: {len(inconsistency)}"
    )

    return inconsistency[
        ["name", "title", "year", "active_start", "album_release_date"]
    ].sort_values(by="name")


def find_tracks_before_birth(df):
    """
    Find tracks released before the artist's birth date.
    """
    tracks_before_birth = df[df["year"] < df["birth_date"].dt.year]

    print(
        f"Number of tracks released before artist's birth: {len(tracks_before_birth)}"
    )
    return tracks_before_birth[
        ["title", "year", "birth_date", "album_release_date", "name"]
    ]


def find_tracks_before_album(df):
    """
    Find tracks released before their album release (excluding singles).
    """
    tracks_before_album = df[
        (df["year"] < df["album_release_date"].dt.year) & (df["album_type"] != "single")
    ]

    print(
        f"Tracks released before the album (excluding singles): {len(tracks_before_album)}"
    )
    return tracks_before_album[
        ["title", "year", "album", "album_release_date", "album_type", "name"]
    ]

def check_skewness(df, cols):
    """
    Calculates mean, median, and skewness for specified columns
    and classifies the skew type based on standard thresholds.
    """
    # Calculate aggregated statistics
    skew_results = df[cols].agg(['mean', 'median', 'skew']).T
    skew_results.columns = ['mean', 'median', 'skewness_value']

    # Define internal function for classification
    def classify_skew(skew_value):
        if skew_value > 0.5:
            return "Positive (Right Skew)"
        elif skew_value < -0.5:
            return "Negative (Left Skew)"
        else:
            return "Symmetric"

    # Apply classification
    skew_results['skew_type'] = skew_results['skewness_value'].apply(classify_skew)

    # Sort by skewness value for better readability
    return skew_results.sort_values(by='skewness_value', ascending=False)
