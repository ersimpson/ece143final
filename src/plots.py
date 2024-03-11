import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def save_bar_plot(filename, df, x, y, title):
    """Create a bar plot from a dataframe and save to disk as a png.

    Args:
        filename (str): The filename of the plot image to save.
        df (DataFrame): A DataFrame of the data to plot as a bar graph.
        x (str): The column name or index of the DataFrame to group x axis
            values in the plot by.
        y (str): The column name of the DataFrame to plo the y axis
            values by.
        title (str): The title of the overall plot.
    
    Returns:
        None
    """
    assert isinstance(filename, str)
    assert isinstance(df, pd.DataFrame)
    df = df.reset_index()
    col_labels = [f"({i}) - {label}" for i, label in enumerate(df[x])]
    with sns.axes_style("whitegrid"):
        ax = sns.barplot(df, x=df.index, y=y, errorbar=None, hue=col_labels, palette=["#7BB594"])
        ax.set_ymargin(0.1)
        ax.set_title(title)
        ax.set_xlabel(x)
        for bar in ax.containers:
            ax.bar_label(bar)
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))  
        plt.savefig(filename, bbox_inches="tight")
        plt.close()