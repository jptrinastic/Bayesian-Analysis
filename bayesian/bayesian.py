# Compilation of classes to demonstrate Bayesian statistical methods

# Libraries
import sys
import math
import numpy as np
import pandas as pd
import scipy.stats as sps
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

class BayesBasic:
    """
    Provides primary methods to for conducting Bayesian analyses.
    Individual likelihood functions are contained in other child classes.
    """
    
    def __init__(self, hypotheses, priorName='uniform',
                 mean=0, stdev=1, alpha=1, peak=5):
        """
        Parameters
        ----------
        hypotheses: numpy array
            array of hypotheses
        prior: string
            prior type to call to initialize below
        mean: float
            value used as mean of gaussian prior
        stdev: float
            standard deviation for gaussian prior
        alpha: float
            value used for exponential decay
        peak: float
            value for peak of triangular distribution
        """
        
        # If trying to instantiate base class, raise exception
        if type(self) is BayesBasic:
            raise Exception("BayesBasic is an abstract class and"\
                            " cannot be instantiated directly."\
                            " Use inherited classes instead.")
        
        # Initialize hypotheses
        self.hypotheses = hypotheses
        minHyp = np.min(self.hypotheses)
        maxHyp = np.max(self.hypotheses)
        
        # Initialize parameters
        self.priorName = priorName
        self.mean = mean
        self.stdev = stdev
        self.alpha = alpha
        self.peak = peak

        # Search for prior and calculate
        dictPrior = {'uniform': np.ones(np.shape(self.hypotheses)[0]),
                     'exponential': np.exp(-self.hypotheses/self.alpha),
                     'gaussian': np.exp(-((self.hypotheses - self.mean)**2)/
                                        (2*(self.stdev**2))),
                     # Triangular: 0.1 added to distribution ends so nonzero probability
                     'triangular': np.where(self.hypotheses<=self.peak,
                                            (2/(maxHyp-minHyp)-minHyp)/(self.peak-minHyp)*self.hypotheses-\
                                            ((2/(maxHyp-minHyp)-minHyp)/(self.peak-minHyp))*minHyp+0.1,
                                            (minHyp-2/(maxHyp-minHyp))/(maxHyp-self.peak)*self.hypotheses-\
                                            ((minHyp-2/(maxHyp-minHyp))/(maxHyp-self.peak))*maxHyp+0.1)
                    }
        
        if self.priorName in dictPrior:
            prior = dictPrior[self.priorName]
        else:
            sys.exit('Cannot recognize prior type.')
        
        # Normalize prior and initialize as class variable
        prior = self._normalize(prior)
        self.prior = prior
        
        # Initiate current probabilities as prior
        self.current = prior
        
        # Initialize posterior as 2d array so that each iteration appends
        # First row of posterior array is the prior (before 1st iteration)
        self.posterior = self.prior.reshape(1,-1)
        
        # Initialize data as array so that each iteration apppends
        self.data = np.array([])
    
    def iterate(self, data):
        """
        Iterate prior to posterior distribution using input data.
        
        Parameters
        ----------
        data: numpy array (of same length as hyp, prior)
            input data for current iteration
        
        Returns
        -------
        None; updates self.current
            
        """
        
        # Append data to self.data
        self.data = np.append(self.data, data)
        
        for i, d in enumerate(data):
            update = self.current*self.likelihood(d)
            self.current = self._normalize(update)
            self.posterior = np.concatenate((self.posterior,[self.current]))
        
        print(str(len(data)) + " iterations completed!")
        
        return None
    
    def cumulative_distribution(self, dist='current'):
        """
        Plots cumulative distribution function of input
        probability distribution.
        
        Parameters
        ----------
        dist: string
            distribution to convert to cdf
            Options: current, prior, posterior
            
        Returns
        -------
        cdf: numpy array
        """
        
        dictDist = {'current': np.cumsum(self.current),
                    'prior': np.cumsum(self.prior),
                    'posterior': np.cumsum(self.posterior, axis=1)
                   }
        
        cdf = dictDist[dist]
        
        return cdf
    
    def credible_interval(self, distType='current', interval=(0.025, 0.975)):
        """
        Calculates credible interval for any probability distribution
        given input interval for cdf.
        
        Parameters
        ----------
        distType: string
            distribution for which to calculate credible interval
            Options: current, prior, posterior
        interval: tuple of floats
            percentiles from cumulative distribution function used
            to calculate credible interval
        
        Returns
        -------
        ci: list of tuples (float-type)
        """
        
        # Calculate cdf to use for credible interval
        distCred = self.cumulative_distribution(dist=distType)
               
        # Prior and Current credible intervals
        if (distType=='current' or distType=='prior'):
            minCred = self.hypotheses[np.where((distCred-interval[0])>0)[0].min()]
            maxCred = self.hypotheses[np.where((distCred-interval[1])>0)[0].min()]
            ci = [(minCred, maxCred)]

        # Posterior: all iterations credible intervals
        else:
            ci = []
            for i, row in enumerate(distCred):
                minCred = self.hypotheses[np.where((distCred[i]-interval[0])>0)[0].min()]
                maxCred = self.hypotheses[np.where((distCred[i]-interval[1])>0)[0].min()]
                ci.append((minCred, maxCred))

        return ci
    
    def plot_cdf(self, distType='posterior', plotType='line', figSize=(5,4)):
        """
        Plots cumulative distribution for various inputs.
        
        Parameters
        ----------
        distType: string
            Type of distribution to plot
        plotType: string
            Determines type of plot using if/then's below.
        figSize: tuple of integers
            Provides figure size for matplotlib figure method.
        
        Returns
        -------
        None: creates plot
        """
        
        # Calculate cdf to plot
        distToPlot = self.cumulative_distribution(dist=distType)
        
        # Create figure
        fig = plt.figure(figsize=figSize)
        
        # Create colormap
        colors = cm.rainbow(np.linspace(0, 1, len(distToPlot)))

        # Determine plot type
        if plotType=='line':
            plt.plot(self.hypotheses, distToPlot.T)
        elif plotType=='bar':
            for row, co in zip(distToPlot, colors):
                plt.bar(self.hypotheses, row, width=0.25,
                        align='center', alpha=0.5, color=co)
        elif plotType=='point':
            for row, co in zip(distToPlot, colors):
                plt.scatter(self.hypotheses, row,
                            alpha=1.0, color=co)
        else:
            sys.exit('Plot type not recognized.')

        plt.legend(np.arange(np.shape(distToPlot)[0]),
                   loc='center left',
                   bbox_to_anchor=(1,0.5),
                   title='Iteration')
        plt.xlabel('Hypotheses', fontsize=14)
        plt.ylabel('Probability', fontsize=14)
        plt.ticklabel_format(useOffset=False)
        
        # If less than 10 hypotheses, treat xticks as categorical
        if len(self.hypotheses) < 20:
            plt.xticks(self.hypotheses)
        
        return None
    
    def plot_posteriors(self, plotType='line',
                        plotEvery=1, figSize=(5,4)):
        """
        Plots all posterior iterations using matplotlib.
        
        Parameters
        ----------
        plotType: string
            Determines type of plot using if/then's below.
        plotEvery: int
            indicates iteration interval for plotting
            posterior distributions
            (e.g. 2 skips every other iteration)
        figSize: tuple of integers
            Provides figure size for matpotlib figure method.
            
        Returns
        -------
        None: creates plot
        """
        
        # Create figure
        fig = plt.figure(figsize=figSize)
        
        # Create colormap
        colors = cm.rainbow(np.linspace(0, 1, len(self.posterior)))

        # Determine plot type
        if plotType=='line':
            plt.plot(self.hypotheses,
                     self.posterior[0::plotEvery,:].T)
        elif plotType=='bar':
            for row, co in zip(self.posterior, colors):
                plt.bar(self.hypotheses, row, width=0.25,
                        align='center', alpha=0.5, color=co)
        elif plotType=='point':
            for row, co in zip(self.posterior, colors):
                plt.scatter(self.hypotheses, row,
                            alpha=1.0, color=co)
        else:
            sys.exit('Plot type not recognized.')

        #np.arange(np.shape(bCoin.posterior[0::10,:].T)[1])*10
        plt.legend(np.arange(np.shape(self.posterior[0::plotEvery,:].T)[1])*plotEvery,
                   loc='center left',
                   bbox_to_anchor=(1,0.5),
                   title='Iteration')
        plt.xlabel('Hypotheses', fontsize=14)
        plt.ylabel('Probability', fontsize=14)
        plt.ticklabel_format(useOffset=False)
        
        # If less than 10 hypotheses, treat xticks as categorical
        if len(self.hypotheses) < 20:
            plt.xticks(self.hypotheses)
            
        return None
            
    
    # Private Methods
    
    def _normalize(self, inp):
        """
        Normalize the product of likelihood and prior.
        
        Parameters
        ----------
        inp: numpy array
            normalize array using sum
        """
        
        return inp/inp.sum()
        
class BayesCookie(BayesBasic):
    """
    Provides likelihood function for basic cookie example
    from Downey's Think Bayes book.
    Only Vanilla (V) and Chocolate(C) cookies in jars.
    """
    
    def __init__(self, hypotheses, proportions, priorName='uniform',
                 mean=0, stdev=1, alpha=1, peak=5):
        """
        Parameters
        ----------
        proportions: numpy array
            Same length as hypotheses, gives V proportion in each bowl.
        """
        
        # Initialize parent class
        super().__init__(hypotheses, priorName,
                         mean, stdev, alpha, peak)
        
        # Initialize proportions of V to C cookies in each bowl
        self.proportions = proportions
    
    def likelihood(self, inData):
        """
        Inverse ikelihood function.
        
        Parameters
        ----------
        inData: float
            input data for current iteration
            
        Returns
        -------
        likelihood: float
        """
        
        lh = np.zeros(len(self.hypotheses))
        
        if inData == 'V':
            lh = self.proportions
        elif inData == 'C':
            lh = 1 - self.proportions
        else:
            sys.exit('Input cookie type not recognized.')
            
        return lh
        
class BayesMandM(BayesBasic):
    """
    Likelihood function for M&M problem.
    Quite unique and unlikely to be used in other applications.
    """
    
    def likelihood(self, inData):
        """
        Parameters
        ----------
        inData: tuple of strings
            pair of M&Ms picked (one from each bag)
        
        Returns
        -------
        likelihood: float
        """
        
        # Initialize likelihood array
        lh = np.zeros(len(self.hypotheses))
        
        # Define probabilities of getting each color
        dict1994 = {'Bl': 0.00, 'Br': 0.30, 'Ye': 0.20,
                    'Re': 0.20, 'Gr': 0.10, 'Or': 0.10,
                    'Ta': 0.10}
        dict1996 = {'Bl': 0.24, 'Br': 0.13, 'Ye': 0.14,
                    'Re': 0.13, 'Gr': 0.20, 'Or': 0.16,
                    'Ta': 0.00}
        
        # Likelihood is 'and' probability of picking each color
        # Always two bags, so always two elements
        lh[0] = dict1994[inData[0]]*dict1996[inData[1]]
        lh[1] = dict1996[inData[0]]*dict1994[inData[1]]

        return lh
    
class BayesMonty(BayesBasic):
    """
    Likelihood function for Monty Hall problem.
    Quite unique and unlikely to be used in other applications.
    """
    
    def likelihood(self, inData):
        """
        Parameters
        ----------
        inData: tuple of ints
            First int: door contestant opens
            Second int: door host opens
        
        Returns
        -------
        likelihood: float
        """
        
        # Initialize likelihood array
        lh = np.zeros(len(self.hypotheses))
        
        if len(lh) != 3:
            sys.exit("Not correct number of hypotheses (3) for Monty Hall problem!")
        
        # Step through each door's likelihood
        # -Door 1: Given the hypothesis that the car is behind Door 1,
        # there's a 50/50 chance the host opens Door 2.
        lh[0] = 1/2
        
        # -Door 2: Given that the car is behind Door 2,
        # the host cannot open Door 2, so will always open Door 3.
        lh[1] = 0
        
        # -Door 3: Given that the car is behind Door 3,
        # the host must open Door 2, otherwise the car would be revealed.
        lh[2] = 1
        
        return lh

class BayesInverse(BayesBasic):
    """
    Provides inverse likelihood function applicable to
    several examples.
    """
        
    def likelihood(self, inData):
        """
        Inverse ikelihood function.
        
        Parameters
        ----------
        inData: float
            input data for current iteration
            
        Returns
        -------
        likelihood: float
        """
        
        lh = np.zeros(len(self.hypotheses))
        
        for i, hyp in enumerate(self.hypotheses):
            if inData > hyp:
                lh[i] = 0
            else:
                lh[i] = 1.0/hyp
            
        return lh

class BayesBinomial(BayesBasic):
    """
    Provides likelihood function for Bernoulii processes that
    lead to binomial distribution (e.g. coin flips).
    Assumes:
    1) hypotheses are written as percents, not decimals
    2) hypotheses are represent the probability of
    the 1-labeled event occurring.
    """
    
    def likelihood(self, inData):
        """
        Likelihood function for Bernoulli process.
        Assumes that 
        
        Parameters
        ----------
        inData: float
            input data for current iteration
            
        Returns
        -------
        likelihood: float
        """
        
        lh = np.zeros(len(self.hypotheses))
        
        if inData==1:
            lh = self.hypotheses/100.0
        else:
            lh = (100 - self.hypotheses)/100.0
        
        return lh
        
class BayesPaint(BayesBasic):
    """
    Provides likelihood function for two-dimensional
    paintball problem.
    Assumes:
    1) hypotheses are written as 2D numpy array;
    2) probability of a shot position based on speed
    (change in x-position w/r/t angle of shooter)
    """
    
    def likelihood(self, inData):
        """
        Likelihood function for 2D paintball process.
        Assumes that 
        
        Parameters
        ----------
        inData: float
            input data for current iteration
            
        Returns
        -------
        likelihood: float
        """
        
        lh = np.zeros(len(self.hypotheses))
        
        # Calculation possible locations (xs) from hypotheses
        locs = list(set(self.hypotheses[:,0]))
        
        # Loop through all hypotheses
        for i, row in enumerate(self.hypotheses):
            # Define a, b position for given hypothesis
            a, b = row
            
            # Then, create pmf for x given a, b
            # - calculate angle for each x
            thetas = np.arctan((locs-a)/b)
            # - calculate probability using speed 1/(dx/dtheta)
            probs = 1.0 / (b / (np.cos(thetas)*np.cos(thetas)))
            probs = self._normalize(probs)
            
            #Then, likelihood is probability of inData
            pos = np.where(locs==inData)[0]
            lh[i] = probs[pos]
               
        return lh        

