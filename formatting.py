from matplotlib.ticker import StrMethodFormatter
from cycler import cycler

def pltSetup(plt):
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    plt.style.use({'figure.facecolor':'white'})
    plt.rcParams['ytick.minor.visible']=True 
    plt.rcParams['xtick.minor.visible']=True
    plt.rcParams['axes.linewidth']=1
    plt.rcParams['axes.prop_cycle'] = cycler(color='bgrcmyk')

def niceplot(ax, font_override=12.):
    """ Simple function to format plots in a way that makes them paper approprate.
    Arguments:
        plt (plotting object) : the plotting axis.
        font_override (float) : the fontsize.
    Returns, None.   
    """

    ax.locator_params(axis='x', nbins=5) # Number of axis labels
    ax.locator_params(axis='y', nbins=5)
    ax.minorticks_on()

    ax.tick_params(axis = "x", which='minor', length=3, width=1., direction='in')
    ax.tick_params(axis="x", length = 5., width=1.5, direction="in", labelsize = font_override)
    ax.xaxis.set_ticks_position('both')

    ax.tick_params(axis = "y", which='minor', length=3, width=1., direction='in')
    ax.tick_params(axis="y", length = 5., width=1.5, direction="in", labelsize = font_override) 
    ax.yaxis.set_ticks_position('both')

    #plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}')) # No decimal places
    #plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}')) # 2 decimal places


