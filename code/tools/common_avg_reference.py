"""
common_avg_reference function   
"""
def common_avg_reference(data_arg):
    """Thus function puts data in common average response

    Args:
        data_arg (pandas.DataFrame): Pandas dataframe with channels in columns

    Returns:
        pandas.DataFrame: Common averaged data frame
    """
    data_arg = data_arg.subtract(data_arg.mean(axis=1), axis=0)
    return data_arg
