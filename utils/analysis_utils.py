"""
A set of functions that are used across the notebook for analysis and visualization.
"""

from collections import Counter

import pandas as pd
from pandas.io.formats.style import Styler
from spacy.tokens.doc import Doc

# Partial import because it will be used in a notebook and things get easier this way
from .constants import TOXIC, NON_TOXIC, TWITTER, NEWS

# --------------------- ANALYSIS METHODS --------------------- #


def get_norm_crosstab(dataset_df: pd.DataFrame, rows_label: str, columns_label: str) -> pd.DataFrame:
    """
    Gets the cross distribution of two columns in a DataFrame.

    Parameters:
    df (DataFrame): The DataFrame to be analyzed.

    Returns:
    DataFrame: A DataFrame with the cross distribution of labels and prefixes.
    """
    # Assert both columns are in the DataFrame
    assert rows_label in dataset_df.columns and columns_label in dataset_df.columns, \
        f"Column {rows_label} or {columns_label} not in DataFrame."

    # Get the cross distribution dataframe
    return pd.crosstab(index=dataset_df[rows_label], columns=dataset_df[columns_label], normalize='all')


def get_word_freq_bias(word_freqs: dict[str, Counter]) -> dict[str, float]:
    """
    Returns the percentage of bias of each word in the word_freqs dictionary.
    :param word_freqs:
    :return:
    """

    assert isinstance(word_freqs, dict) and all(isinstance(value, Counter) for value in word_freqs.values()), \
        f"word_freqs must be a dictionary of Counters."

    # Calculate bias percentages
    total_freqs = Counter()
    for counter in word_freqs.values():
        total_freqs.update(counter)

    # Normalize each word frequency by the total number of words
    biases = {label: {word: freq / total_freqs[word] for word, freq in label_freqs.items()}
                            for label, label_freqs in word_freqs.items()}

    return biases


def get_word_freqs(processed_words: dict[str, Doc|dict]) -> dict[str, Counter|dict]:
    """
    Recursive for being agnostic of nesting level.
    :param processed_words:
    :return:
    """
    def _word_freq_from_list_of_docs(docs: list[Doc]) -> Counter:
        """
        Counts the number of occurrences of each lemma in a list of Doc objects.
        """
        assert all(isinstance(doc, Doc) for doc in docs), "Final word_freq nesting level must be a list of Docs or str (words)."
        return Counter(word.lemma_.lower() for doc in docs for word in doc
                       if not word.is_stop and not word.is_punct and not word.is_space)

    assert isinstance(processed_words, dict), "processed_words is expected to be a dictionary of Docs " \
                                              "(with any nesting level)."

    # Do it recursively if the value is a dictionary to be agnostic of nesting level
    return {key: get_word_freqs(processed_words=value) if isinstance(value, dict) else _word_freq_from_list_of_docs(docs=value)
            for key, value in processed_words.items()}


# --------------------- VISUALIZATION METHODS --------------------- #

def visualize_norm_crosstab(cross_dist_df: pd.DataFrame, title: str,
                                    rows_name: str, columns_name: str) -> Styler:
    """
    Visualizes the cross distribution of labels and prefixes for a given dataset, including totals
    that reflect the proportion of each label and each prefix independently.
    """

    # Copy the dataframe to avoid unwanted modifications
    cross_dist_with_totals = cross_dist_df.copy()
    # Create a new column with the totals of each row
    cross_dist_with_totals[f"TOTAL ({columns_name})"] = cross_dist_with_totals.sum(axis=1)

    # Create a new row with the totals of each column
    totals_row = cross_dist_with_totals.sum(axis=0)
    totals_row.name = f"TOTAL ({rows_name})"
    cross_dist_with_totals = pd.concat([cross_dist_with_totals, totals_row.to_frame().T], axis=0)

    # Rename the top left cell
    cross_dist_with_totals = cross_dist_with_totals.rename_axis(f"{columns_name} / {rows_name}", axis="columns")

    # Estilizar el DataFrame
    return cross_dist_with_totals.style.format("{:.2%}").set_table_styles(
        [{'selector': 'th',
          'props': [('background-color', '#f4f4f4'),
                    ('color', 'black'),
                    ('font-weight', 'bold'),
                    ('font-family', 'monospace'),
                    ('text-align', 'center')]},
         {'selector': 'td',
          'props': [('text-align', 'center')]}]
    ).set_properties(**{'background-color': 'white', 'color': 'black'}).set_caption(caption=title)


def style_dataframe(df: pd.DataFrame, num_rows: int = 5) -> pd.DataFrame:
    """
    Apply a basic style to the first 'num_rows' rows of a DataFrame.

    Parameters:
    df (DataFrame): The DataFrame to be styled.
    num_rows (int): Number of rows to display. Defaults to 5.

    Returns:
    Styler: A DataFrame with applied styles.
    """
    return df.head(num_rows).style.set_table_styles(
        [{'selector': 'th',
          'props': [('background-color', '#f4f4f4'),
                    ('color', 'black'),
                    ('font-weight', 'bold'),
                    ('text-align', 'center')]},
         {'selector': 'td',
          'props': [('text-align', 'left')]}]).hide(axis='index')


def visualize_word_freqs_by_toxicity_and_origin(word_freqs: dict[str, dict[str, Counter]], max_rows: int = 100) \
            -> tuple[dict[str, Counter], pd.DataFrame]:

    assert isinstance(word_freqs, dict) and all(isinstance(value, dict) for value in word_freqs.values()), \
        f"word_freqs must be a dictionary of dictionaries of Counters."
    assert all(key in (TOXIC, NON_TOXIC) for key in word_freqs), \
        f"word_freqs must have only {TOXIC} and {NON_TOXIC} keys."
    assert all(all(key in (TWITTER, NEWS) for key in origins) for origins in word_freqs.values()), \
        f"word_freqs must have only {TWITTER} and {NEWS} keys in the second level."

    # We want to show the global bias of words, independently of the origin
    global_toxicity_counts = {TOXIC: word_freqs[TOXIC][TWITTER] + word_freqs[TOXIC][NEWS],
                              NON_TOXIC: word_freqs[NON_TOXIC][TWITTER] + word_freqs[NON_TOXIC][NEWS]}

    global_toxicity_freqs = get_word_freq_bias(word_freqs=global_toxicity_counts)

    dataframes = {TOXIC: [], NON_TOXIC: []}

    for label in word_freqs:
        for origin in word_freqs[label]:
            # Generate the column with word and counts
            partial_df = pd.DataFrame(word_freqs[label][origin].most_common(max_rows), columns=['word', 'count'])
            # Add a new column with ratio
            partial_df['ratio'] = partial_df['word'].apply(func=lambda x: round(global_toxicity_freqs[label][x] * 100, ndigits=2))
            # Sort by ratio
            partial_df.sort_values(by='ratio', ascending=False, inplace=True)
            # Reset index to make sure they are all placed side to side
            partial_df.reset_index(drop=True, inplace=True)

            # Add the DataFrame to the corresponding list
            dataframes[label].append(partial_df)

    # Concatenar los DataFrames en df
    dataframes = {label: pd.concat(dataframes[label], keys=[origin for origin in word_freqs[label]], axis=1)
                    for label in dataframes}

    # Concatenar todos los DataFrames para crear el DataFrame final con MultiIndex
    bag_of_words = pd.concat(dataframes, axis=1)

    # Reordenar columnas
    columns_order = []
    for label in bag_of_words.columns.levels[0]:
        for origin in bag_of_words.columns.levels[1]:
            columns_order.extend([(label, origin, 'word'),
                                 (label, origin, 'count'),
                                 (label, origin, 'ratio')])

    bag_of_words = bag_of_words.reindex(columns=pd.MultiIndex.from_tuples(columns_order))

    return global_toxicity_freqs, style_dataframe(bag_of_words, num_rows=max_rows)


def format_text_col_df(row: pd.Series):
    """
    Formats a single row to create a string containing the text in different languages.
    If 'id-type' is 'url', it prefixes the text with the URL.
    If 'id-type' is 'twitter', it prefixes the text with '[twitter]'.
    It handles missing values in 'english' and 'french'.
    """
    # Determine the prefix based on 'id-type'
    prefix = f"[{row['id']}]" if row['id-type'] == 'url' else '[twitter]'

    # Format the text for each language
    formatted_texts = []
    if pd.notna(row['text']):
        formatted_texts.append(f"{prefix} {row['text']}")
    if pd.notna(row['english']):
        formatted_texts.append(f"{prefix} {row['english']}")
    if pd.notna(row['french']):
        formatted_texts.append(f"{prefix} {row['french']}")

    # Join the texts from different languages
    return formatted_texts


def get_dataset_ready(df: pd.DataFrame):
    """
    Apply a formatting function to each row of a DataFrame and expand the labels to match the length of the formatted texts.

    Parameters:
    df (DataFrame): The original DataFrame.

    Returns:
    DataFrame: A new DataFrame with formatted texts and expanded labels.
    """
    # Apply the formatting function to each row and concatenate the results
    all_texts = df.apply(format_text_col_df, axis=1).sum()

    # Expand the labels to match the length of all_texts
    expanded_labels = df.loc[df.index.repeat(df.apply(lambda x: len(format_text_col_df(x)), axis=1))]['label']

    # Create a new DataFrame with the formatted texts and the expanded labels
    formatted_df = pd.DataFrame({'text': all_texts, 'label': expanded_labels.reset_index(drop=True)})

    return formatted_df
