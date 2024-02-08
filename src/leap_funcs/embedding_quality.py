import numpy as np
import scipy as sp

def analyze_embedding(emb:dict={}, metrics:dict={}, verbose=False):
    """
    Analyze the quality of an embedding by computing the following metrics:
    """
    

    _comp_all = False
    if len(emb) == 0:
        raise ValueError("Empty embedding. Please provide a valid embedding.")
    if len(metrics) == 0:
        _comp_all = True
    
    emb_chain_lengths = np.array([len(v) for v in emb.values()])
    
    result = {}
    result['num. chains'] = emb_chain_lengths.shape[0]
    result['num. qubits'] = len(set(inner for outer in emb.values() for inner in outer))
    result['metrics chain_lengths'] = {}
    

    if _comp_all or 'mean' in metrics.keys():
        """scipy.stats.tmean(a, limits=None, inclusive=(True, True), axis=None)
            Compute the trimmed mean.
            This function finds the arithmetic mean of given values, ignoring values outside the given limits."""
        key, val = 'mean', sp.stats.tmean(emb_chain_lengths, limits=None)
        result['metrics chain_lengths'][key] = val
        if verbose: print(key+':', val)
    
    if _comp_all or 'min_max' in metrics.keys():
        """ simply minimum and maximum"""
        key, val = 'min_max', [np.min(emb_chain_lengths), np.max(emb_chain_lengths)]
        result['metrics chain_lengths']['min max'] = val
        if verbose: print(key+':', val)

    if _comp_all or 'median' in metrics.keys():
        """median is 50% expectile, see expectile for more info"""
        key, val = 'median', np.array(sp.stats.expectile(emb_chain_lengths, alpha=0.5))
        result['metrics chain_lengths']['median'] = val
        if verbose: print(key+':', val)

    if _comp_all or 'mode' in metrics.keys():
        """scipy.stats.mode(a, axis=0, nan_policy='propagate', keepdims=False)
            Return an array of the modal (most common) value in the passed array.
            If there is more than one such value, only one is returned. The bin-count for the modal bins is also returned. """
        key, val = 'mode', sp.stats.mode(emb_chain_lengths)
        result['metrics chain_lengths']['mode'] = val
        if verbose: print(key+':', val)

    if _comp_all or 'var' in metrics.keys():
        """scipy.stats.tvar(a, limits=None, inclusive=(True, True), axis=0, ddof=1)
            Compute the trimmed variance.
            This function computes the sample variance of an array of values, while ignoring values which are outside of given limits."""
        key, val = 'var', sp.stats.tvar(emb_chain_lengths, limits=None)
        result['metrics chain_lengths']['var'] = val
        if verbose: print(key+':', val)

    if _comp_all or 'std' in metrics.keys():
        """scipy.stats.tstd(a, limits=None, inclusive=(True, True), axis=0, ddof=1)
            Compute the trimmed sample standard deviation.
            This function finds the sample standard deviation of given values, ignoring values outside the given limits."""
        key, val = 'std', sp.stats.tstd(emb_chain_lengths, limits=None)
        result['metrics chain_lengths']['std'] = val
        if verbose: print(key+':', val)

    if _comp_all or 'variation' in metrics.keys():
        """scipy.stats.variation(a, axis=0, nan_policy='propagate', ddof=0, *, keepdims=False)[source]
            Compute the coefficient of variation.
            The coefficient of variation is the standard deviation divided by the mean. This function is equivalent to: np.std(x, axis=axis, ddof=ddof) / np.mean(x)"""
        key, val = 'variation', sp.stats.variation(emb_chain_lengths)
        result['metrics chain_lengths']['variation'] = val
        if verbose: print(key+':', val)

    if _comp_all or 'iqr' in metrics.keys():
        """scipy.stats.iqr(x, axis=None, rng=(25, 75), scale=1.0, nan_policy='propagate', interpolation='linear', keepdims=False)[source]
            Compute the interquartile range of the data along the specified axis.
            The interquartile range (IQR) is the difference between the 75th and 25th percentile of the data. It is a measure of the dispersion similar to standard deviation or variance, but is much more robust against outliers [2]."""
        key, val = 'iqr', sp.stats.iqr(emb_chain_lengths)
        result['metrics chain_lengths']['iqr'] = val
        if verbose: print(key+':', val)

    if _comp_all  or 'bayes_mvs' in metrics.keys():
        """scipy.stats.bayes_mvs(data, alpha=0.9)[source]
            Bayesian confidence intervals for the mean, var, and std."""
        _mean, _var, _std = sp.stats.bayes_mvs(emb_chain_lengths)
        result['metrics chain_lengths']['bayes_mvs'] = [_mean, _var, _std]
        if verbose: print('bayes_mvs:', _mean, _var, _std)

    if _comp_all or 'entropy' in metrics.keys():
        """scipy.stats.entropy(pk, qk=None, base=None, axis=0)
            Calculate the Shannon entropy/relative entropy of given distribution(s).
            If only probabilities pk are given, the Shannon entropy is calculated as H = -sum(pk * log(pk)).
            If qk is not None, then compute the relative entropy D = sum(pk * log(pk / qk)). This quantity is also known as the Kullback-Leibler divergence.
            This routine will normalize pk and qk if they dont sum to 1."""
        key, val = 'entropy', sp.stats.entropy(emb_chain_lengths)
        result['metrics chain_lengths']['entropy'] = val
        if verbose: print(key+':', val)

    if _comp_all or 'differential_entropy' in metrics.keys():
        """scipy.stats.differential_entropy(values, *, window_length=None, base=None, axis=0, method='auto')
            Given a sample of a distribution, estimate the differential entropy.
            Several estimation methods are available using the method parameter. By default, a method is selected based the size of the sample."""
        key, val = 'differential_entropy', sp.stats.differential_entropy(emb_chain_lengths)
        result['metrics chain_lengths']['differential_entropy'] = val
        if verbose: print(key+':', val)

    if _comp_all or 'kurtosis' in metrics.keys():
        """ scipy.stats.kurtosis(a, axis=0, fisher=True, bias=True, nan_policy='propagate', *, keepdims=False)
            Compute the kurtosis (Fisher or Pearson) of a dataset.
            Kurtosis is the fourth central moment divided by the square of the variance. If Fisher’s definition is used, then 3.0 is subtracted from the result to give 0.0 for a normal distribution.
            If bias is False then the kurtosis is calculated using k statistics to eliminate bias coming from biased moment estimators"""
        key, val = 'kurtosis', sp.stats.kurtosis(emb_chain_lengths)
        result['metrics chain_lengths']['kurtosis'] = val
        if verbose: print(key+':', val)

    if _comp_all or 'moment' in metrics.keys():
        """scipy.stats.moment(a, moment=1, axis=0, nan_policy='propagate', *, center=None, keepdims=False)
            Calculate the nth moment about the mean for a sample.
            A moment is a specific quantitative measure of the shape of a set of points. It is often used to calculate coefficients of skewness and kurtosis due to its close relationship with them."""
        key, val = 'moment', np.array([sp.stats.moment(emb_chain_lengths, moment=i) for i in [1, 2, 3, 4]])
        result['metrics chain_lengths']['moments mean 1st 2nd 3rd 4th'] = val
        if verbose: print(key+':', val)

    if _comp_all or 'expectile' in metrics.keys():
        """scipy.stats.expectile(a, alpha=0.5, *, weights=None)
            Compute the expectile at the specified level.
            Expectiles are a generalization of the expectation in the same way as quantiles are a generalization of the median. The expectile at level alpha = 0.5 is the mean (average)."""
        key, val = 'expectile', np.array([sp.stats.expectile(emb_chain_lengths, alpha=i) for i in [0.125, 0.25, 0.5, 0.75, 0.875]])
        result['metrics chain_lengths']['expectiles .125 .25 .5 .75 .875'] = val
        if verbose: print(key+':', val)

    if _comp_all or 'skew' in metrics.keys():
        """scipy.stats.skew(a, axis=0, bias=True, nan_policy='propagate', *, keepdims=False)
            Compute the sample skewness of a data set.
            For normally distributed data, the skewness should be about zero. For unimodal continuous distributions, a skewness value greater than zero means that there is more weight in the right tail of the distribution. The function skewtest can be used to determine if the skewness value is close enough to zero, statistically speaking."""
        key, val = 'skew', sp.stats.skew(emb_chain_lengths)
        result['metrics chain_lengths']['skew'] = val
        if verbose: print(key+':', val)

    if _comp_all or 'kstat' in metrics.keys():
        """scipy.stats.kstat(data, n=2, *, axis=None, nan_policy='propagate', keepdims=False)
            Return the nth k-statistic (1<=n<=4 so far).
            The nth k-statistic k_n is the unique symmetric unbiased estimator of the nth cumulant kappa_n."""
        key, val = 'kstat', np.array([sp.stats.kstat(emb_chain_lengths, n=i) for i in [1, 2, 3, 4]])
        result['metrics chain_lengths']['kstat 1st 2nd 3rd 4th'] = val
        if verbose: print(key+':', val)
    
    if _comp_all or 'kstatvar' in metrics.keys():
        """scipy.stats.kstatvar(data, n=2, *, axis=None, nan_policy='propagate', keepdims=False)
            Return an unbiased estimator of the variance of the k-statistic."""
        # supported only for 1st and 2nd k-statistic 
        key, val = 'kstatvar', np.array([sp.stats.kstatvar(emb_chain_lengths, n=i) for i in [1, 2]])
        result['metrics chain_lengths']['kstatvar 1st 2nd'] = val
        if verbose: print(key+':', val)

    return result
    