#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here
    cleaned_data = [(x[0],y[0],abs(z[0]-y[0])) for x,y,z in zip(ages, net_worths, predictions)]
    cleaned_data = sorted(cleaned_data, key=lambda k: k[2])
    cleaned_data = cleaned_data[:-9]
    return cleaned_data

