import pandas as pd
import os


class Linear:
    '''Calculates variables for linear regression. first column of 'frame' is taken as x, second - y.'''
    import pandas as pd
    def __init__(self, frame : pd.DataFrame):
        self.data = frame.iloc[:, :2]
        self.X = frame.iloc[:, 0]
        self.Y = frame.iloc[:, 1]
       
        ## means
        self.mean_X = self.X.mean()
        self.mean_Y = self.Y.mean()

        ## slope
        self.slope = sum( [ (x-self.mean_X)*(y-self.mean_Y) for x, y in zip(self.X, self.Y) ] ) / (sum( [(x-self.mean_X)**2 for x in self.X] ))

        ## intercept
        self.intercept = self.mean_Y - (self.slope * self.mean_X)

        ## predicted Y series
        self.Y_pred = pd.Series( [(self.intercept + self.slope * x) for x in self.X] )

        ### rss residual sum of squares
        self.rss = sum( [ (y - yy)**2 for y, yy in zip(self.Y, self.Y_pred) ] )

        ## RSE residual standard error
        self.rse = (self.rss/(self.X.shape[0] - 2))**(1/2)

        self.n = self.X.shape[0]
        ## standard error - mean
        self.SE_mean = self.rse**2/self.n

        ## standard error - intercept
        self.SE_intercept = ( self.rse**2*( 1/self.n+( self.mean_X**2/( sum([ (x-self.mean_X)**2 for x in self.X ]) ) ) ) )**(1/2)

        ## standard error - slope
        self.SE_slope = ( self.rse**2/(sum( [(x-self.mean_X)**2 for x in self.X] )) )**(1/2)

        ## t-statistics
        self.tstat_slope = (self.slope-0)/self.SE_slope
        self.tstat_intercept = (self.intercept-0)/self.SE_intercept

        ## tss total sum of squares
        self.tss = sum([ (y-self.mean_Y)**2 for y in self.Y ])

        ## r^2 statistics
        self.r2 = (self.tss-self.rss)/self.tss

         

    def display(self):
        # print(self.data)
        print("---BEGIN---- x is ")
        print("slope: {}".format(self.slope))
        print("intercept: {}".format(self.intercept))
        print("RSS: {}".format(self.rss))
        print("RSE: {}".format(self.rse))
        print("tss: {}".format(self.tss))
        print("SE mean: {}".format(self.SE_mean))
        print("SE intercept: {}".format(self.SE_intercept))
        print("SE slope: {}".format(self.SE_slope))
        print("t-statistic slope: {}".format(self.tstat_slope))
        print("t-statistic intercept: {}".format(self.tstat_intercept))
        print("R^2 statistic: {}".format(self.r2))
        print("----END----")
        print("confidence interval for B0 [{}, {}]".format(self.intercept-2*self.SE_intercept,self.intercept+2*self.SE_intercept))
        print("confidence interval for B1 [{}, {}]".format(self.slope-2*self.SE_slope,self.slope+2*self.SE_slope))
    
    def prediction(self) -> pd.Series:
        return pd.Series([ self.intercept+self.slope*x for x in self.X ])


if __name__ == '__main__':
    pass
