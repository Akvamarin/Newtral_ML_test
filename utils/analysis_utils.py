"""
A set of functions that are used during the analysis of the dataset, for both
data processing and visualization.
Moved here to avoid cluttering the notebooks with code that is not relevant to the analysis.
"""

from collections import Counter

import pandas as pd
from pandas.io.formats.style import Styler
from spacy.tokens.doc import Doc
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Partial import because it will be used in a notebook and things get easier this way
from .constants import TOXIC, NON_TOXIC, TWITTER, NEWS


# ---------------------------------- ANALYSIS METHODS ---------------------------------- #

def get_norm_crosstab(dataset_df: pd.DataFrame, rows_label: str, columns_label: str) -> pd.DataFrame:
    """
    Returns the normalized cross distribution of two columns in the DataFrame.
    Used for bias analysis.

    :param dataset_df: pd.DataFrame. The DataFrame containing the dataset, must contain both columns.
    :param rows_label: str. The name of the column to be used as rows.
    :param columns_label: str. The name of the column to be used as columns.

    :return: pd.DataFrame. A 2x2 DataFrame with the normalized cross distribution of the two columns.

    :raises AssertionError: If any of the columns is not in the DataFrame.
    """
    # Assert both columns are in the DataFrame
    assert rows_label in dataset_df.columns and columns_label in dataset_df.columns, \
        f"Column {rows_label} or {columns_label} not in DataFrame."

    # Get the cross distribution dataframe
    return pd.crosstab(index=dataset_df[rows_label], columns=dataset_df[columns_label], normalize='all')


def get_word_freq_bias(word_freqs: dict[str, Counter]) -> dict[str, float]:
    """
    Returns the bias of each word in a dictionary of Counters. The bias is calculated as the percentage of
    occurrences of a word in a label over the total number of occurrences of that word in all labels. Those
    labels are the keys of the dictionary.

    :param word_freqs: dict[str, Counter]. A dictionary of Counters, where each Counter contains the number of
    occurrences of each word in a label.

    :return: dict[str, float]. A dictionary of floats, where each float is the bias of a word in a label. For example,
    if the word 'hello' appears 10 times in the label 'toxic' and 20 times in the label 'non-toxic', the dictionary
    will look like: {'toxic': {'hello': 0.33, ...}, 'non-toxic': {'hello': 0.66, ...}}.

    :raises AssertionError: If the word_freqs parameter is not a dictionary of Counters.
    """

    assert isinstance(word_freqs, dict) and all(isinstance(value, Counter) for value in word_freqs.values()), \
        f"word_freqs must be a dictionary of Counters."

    # Calculate total number of occurrences of each word
    total_freqs = Counter()
    for counter in word_freqs.values():
        total_freqs.update(counter)

    # Normalize each word frequency in a label by the total number of occurrences of that word
    biases = {label: {word: freq / total_freqs[word] for word, freq in label_freqs.items()}
                            for label, label_freqs in word_freqs.items()}

    return biases


def get_word_freqs(processed_words: dict[str, list[Doc]|dict]) -> dict[str, Counter|dict]:
    """
    Returns the number of occurrences of each lemma in a dictionary of lists of Docs (avoiding stopwords and punctuation).
    That function is recursive, so it can be used with any nesting level.
    :param processed_words: dict[str, list[Doc]|dict]. A dictionary of lists of Docs, where each Doc contains the
     processed words for a given line of text. The nesting level can be any, as long as the final level is a list of Docs.

    :return: dict[str, Counter|dict]. A dictionary of Counters, where each Counter contains the number of occurrences
    of each lemma in a Doc. The dictionary structure is the same as the input, but the final level is a Counter instead
    of a list of Docs

    :raises AssertionError: If the processed_words is not a dictionary of lists of Docs (with any nesting level).
    """
    def _word_freq_from_list_of_docs(docs: list[Doc]) -> Counter:
        """
        Counts the number of occurrences of each lemma in a list of Doc objects.

        :param docs: list[Doc]. A list of Doc objects.

        :return: Counter. A Counter containing the number of occurrences of each lemma in the given list of Docs.

        :raises AssertionError: If the docs parameter is not a list of Doc objects.
        """
        assert all(isinstance(doc, Doc) for doc in docs), "Final word_freq nesting level must be a list of Docs or str (words)."

        # Group (relevant) lemmas in a Counter
        return Counter(word.lemma_.lower() for doc in docs for word in doc
                       if not word.is_stop and not word.is_punct and not word.is_space)

    assert isinstance(processed_words, dict), "processed_words is expected to be a dictionary of Docs " \
                                              "(with any nesting level)."

    # Do it recursively if the value is a dictionary to be agnostic of nesting level
    return {key: get_word_freqs(processed_words=value) if isinstance(value, dict) else _word_freq_from_list_of_docs(docs=value)
            for key, value in processed_words.items()}


# ---------------------------------- VISUALIZATION METHODS ---------------------------------- #

def visualize_norm_crosstab(cross_dist_df: pd.DataFrame, title: str,
                                    rows_name: str, columns_name: str) -> Styler:
    """
    Visualizes the cross distribution of labels and prefixes for a given dataset, including totals
    that reflect the proportion of each label and each prefix independently.
    cross_dist_df is the expected output of get_norm_crosstab.

    :param cross_dist_df: pd.DataFrame. The cross distribution of labels and prefixes for a given dataset.
    :param title: str. The title of the table.
    :param rows_name: str. The name of the rows label. Same as rows_label parameter in get_norm_crosstab.
    :param columns_name: str. The name of the columns label. Same as columns_label parameter in get_norm_crosstab.

    :return: Styler. A styled DataFrame with the cross distribution of labels and prefixes for a given dataset. Can
    be directly displayed in a notebook.
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

    # Style the dataframe
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
    Crop the DataFrame to the first num_rows rows and apply some styles to it.

    :param df: pd.DataFrame. The DataFrame to be cropped and styled.
    :param num_rows: int. The number of rows to be shown.

    :return: pd.DataFrame. The cropped and styled DataFrame.
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
            -> tuple[dict[str, Counter], Styler]:
    """
    Gets a Styled Dataframe for visualizing the frequency of words by their toxicity and origin (Twitter or News)
     in the dataset. This function creates a styled DataFrame showing the most common words in each category, ordered
     by their bias, along with their counts and global toxicity ratios.

    :param word_freqs: dict[str, dict[str, Counter]]. A nested dictionary where the first key is the toxicity label
    (Toxic/Non-Toxic), and the second key is the origin (Twitter/News). Each entry contains a Counter with
     the word frequencies for that category.
    :param max_rows: int, optional. The maximum number of rows to display for each category.

    :return: tuple[dict[str, Counter], Styler]. Returns a tuple containing:
        1. A dictionary with global toxicity frequencies of words across all origins.
        2. A styled DataFrame showing word frequencies, counts, and global toxicity ratios for each  toxicity label and
         each origin.

    :raises AssertionError: If word_freqs is not a nested dictionary as described or if it contains unexpected keys.
    Expected format is:
     {'toxic': {'twitter': Counter, 'news': Counter}, 'non-toxic': {'twitter': Counter, 'news': Counter}}.
    """


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
            partial_df['ratio'] = partial_df['word'].apply(func=lambda x: global_toxicity_freqs[label][x] * 100)
            # Orden before applying format (there is a 100% value)
            partial_df.sort_values(by='ratio', ascending=False, inplace=True)
            partial_df['ratio'] = partial_df['ratio'].apply("{:.2f}".format)
            # Reset index to make sure they are all placed side to side
            partial_df.reset_index(drop=True, inplace=True)
            # Add the DataFrame to the corresponding list
            dataframes[label].append(partial_df)

    # Concatenate the DataFrames for each label and origin to create a DataFrame with MultiIndex
    dataframes = {label: pd.concat(dataframes[label], keys=[origin for origin in word_freqs[label]], axis=1)
                    for label in dataframes}

    # Concatenate the DataFrames for each label to create a DataFrame with MultiIndex
    bag_of_words = pd.concat(dataframes, axis=1)

    # Reorder the columns
    columns_order = []
    for label in bag_of_words.columns.levels[0]:
        for origin in bag_of_words.columns.levels[1]:
            columns_order.extend([(label, origin, 'word'),
                                 (label, origin, 'count'),
                                 (label, origin, 'ratio')])

    # Reindex the DataFrame with the new order
    bag_of_words = bag_of_words.reindex(columns=pd.MultiIndex.from_tuples(columns_order))

    return global_toxicity_freqs, style_dataframe(bag_of_words, num_rows=max_rows)


def show_toxicity_vs_sentiment_confusion_matrix(toxicity_labels: list[int], sentiment_labels: list[int],
                                                title: str):
    """
   Plots a pseudo-confusion matrix to visualize the relationship between toxicity and sentiment labels in a dataset.
   The function generates a matrix showing the proportion of each sentiment label within each toxicity category.

   :param toxicity_labels: list[int]. A list of integers representing toxicity labels (0 for non-toxic, 1 for toxic).
   :param sentiment_labels: list[int]. A list of integers representing sentiment labels (0 for positive, 1 for neutral,
    2 for negative).
   :param title: str. The title of the plot.


   """
    # Calculate the confusion matrix (not strictly what we usually call confusion matrix)
    cm = confusion_matrix(y_true=toxicity_labels, y_pred=sentiment_labels, normalize='true')
    # Remove the last row and cast it to percentage
    cm = cm[:2, :]*100

    # Plot the confusion matrix
    plt.matshow(cm, cmap=plt.cm.Purples)
    plt.colorbar()
    # Plot the axis labels and the title
    plt.title(title)
    plt.xlabel('Sentiment')
    plt.ylabel('Toxicity')
    # Plot the ticks
    plt.xticks(np.arange(cm.shape[1]), ['Postive', 'Neutral', 'Negative'])
    plt.yticks(np.arange(cm.shape[0]), ['Non Toxic', 'Toxic'])

    # Draw the percentages within each cell
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(x=j, y=i, s=f"{round(cm[i, j], ndigits=1)}%", 
                    ha='center', va='center', color='black')

    plt.show()