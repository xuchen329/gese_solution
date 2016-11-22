import numpy as np
from scipy import stats

def import_data(fname):
    return np.genfromtxt(fname,delimiter=',',dtype=[('number','i8'),('dpr','i8'),('gene','i8'),('strs','i8')],skip_header=1)

# linear regression with 2 independent variables
def linear_regression(x1,x2,y):
    if (len(x1) != len(x2)) or (len(x1) != len(y)):
        raise ValueError('size of x1 x2 and y should be equal')
    ones = np.ones(len(y))
    mxp = np.matrix([ones,x1,x2])
    mx = mxp.transpose()
    mxpmx_inv = np.linalg.inv(np.dot(mxp,mx))
    my = np.matrix(y).transpose()
    # esimators
    mb = mxpmx_inv*mxp*my
    mb_val = np.array(mb).flatten()

    # sum of squared error
    SSE = np.sum(y*y)-mb.transpose()*mxp*my
    dof = len(y)-(2+1)

    s_square = float(SSE)/float(dof)
    se_of_b1 = np.sqrt(s_square*np.array(mxpmx_inv)[1][1])
    se_of_b2 = np.sqrt(s_square*np.array(mxpmx_inv)[2][2])

    t_score_b1 = mb_val[1]/se_of_b1
    t_score_b2 = mb_val[2]/se_of_b2

    p_value_b1 = stats.t.sf(np.abs(t_score_b1),dof)*2
    p_value_b2 = stats.t.sf(np.abs(t_score_b2),dof)*2

    print('coeff. x1:  %.3f, err: %.3f, p-value: %.3f' % (mb_val[1],se_of_b1,p_value_b1))
    print('coeff. x2:  %.3f, err: %.3f, p-value: %.3f' % (mb_val[2],se_of_b2,p_value_b2))

if __name__ == '__main__':

    # load data
    dataobj = import_data('gesedata.csv')

    # multi-variate linear regression of full dataset
    linear_regression(dataobj['gene'],dataobj['strs'],dataobj['dpr'])

    '''Results:
    coeff. x1:  -0.450, err: 0.426, p-value: 0.290
    coeff. x2:  0.658, err: 0.096, p-value: 0.000
    '''
    # >99.9% confidence level to reject H0 of b2==0, extremely strong evidence on the linear correlation between # of stressful events and depression score
    # weak confidence in absence of gene (gene==0) triggering the onset of depression.

    # Now create a mask to look at people brought up in relatively negative environment (stressful life events higher than median of the population)
    mask = dataobj['strs']>np.median(dataobj['strs'])
    # multi-variate linear regression of the sub-dataset
    linear_regression(dataobj['gene'][mask],dataobj['strs'][mask],dataobj['dpr'][mask])

    '''Results:
    coeff. x1:  -1.527, err: 0.697, p-value: 0.029
    coeff. x2:  0.644, err: 0.168, p-value: 0.000
    '''
    # Still extremely strong evidence on the linear correlation between  # of stressful events and depression score
    # Also shows strong evidence (>95%) on the absence of gene (gene==0) triggered high depression score.
