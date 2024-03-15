import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def save_audio_features_hist(filename, audio_df, x, title):
    """Create a histogram of audio characteristics data for track information
    and save to disk as png.

    Args:
        filename (str): The filename of the plot image to save.
        audio_df (DataFrame): A DataFrame of the audio characteristics data to 
            plot as a histogram.
        x (str): The column name or index of the DataFrame to group x axis
            values in the plot by.
        title (str): The title of the overall plot.
    
    Returns:
        None
    """
    with sns.axes_style("whitegrid"):
        ax = sns.histplot(audio_df, x=x, hue="label", stat="density", common_norm=False)
        ax.set_title(title)
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))  
        plt.savefig(filename, bbox_inches="tight")
        plt.close()


def save_bar_plot(filename, df, x, y, title, orient="v"):
    """Create a bar plot from a dataframe and save to disk as a png.

    Args:
        filename (str): The filename of the plot image to save.
        df (DataFrame): A DataFrame of the data to plot as a bar graph.
        x (str): The column name or index of the DataFrame to group x axis
            values in the plot by.
        y (str): The column name of the DataFrame to plot the y axis
            values by.
        title (str): The title of the overall plot.
        orient (str): The orientation of the bar plot.
    
    Returns:
        None
    """
    assert isinstance(filename, str)
    assert isinstance(df, pd.DataFrame)
    assert isinstance(x, str)
    assert isinstance(y, str)
    assert isinstance(title, str)
    assert isinstance(orient, str)
    assert orient in ("h", "v")
    
    df = df.reset_index()

    assert x in df.columns
    assert y in df.columns

    label_col = x
    if orient == "v":
        xlabel, ylabel = x, y
        x, y = df.index, y
        xmargin, ymargin = 0.05, 0.15
    elif orient == "h":
        xlabel, ylabel = y, x
        x, y = y, df.index
        xmargin, ymargin = 0.15, 0.05
    col_labels = [f"({i}) - {label}" for i, label in enumerate(df[label_col])]
    with sns.axes_style("whitegrid"):
        ax = sns.barplot(df, x=x, y=y, errorbar=None, hue=col_labels, palette=["#7BB594"], orient=orient)
        ax.set_xmargin(xmargin)
        ax.set_ymargin(ymargin)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        for bar in ax.containers:
            ax.bar_label(bar)
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))  
        plt.savefig(filename, bbox_inches="tight")
        plt.close()