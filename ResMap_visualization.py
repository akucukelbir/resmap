import numpy as np
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt


def visualize2DPlots(**kwargs):
    # Get inputs
    dataOrig   = kwargs.get('dataOrig',  None)
    minRes     = kwargs.get('minRes', 0.0)
    maxRes     = kwargs.get('maxRes', 0.0)
    resTOTALma = kwargs.get('resTOTALma',  None)
    resHisto   = kwargs.get('resHisto', None)

    # Plot slices of orignal volume
    plotOriginalVolume(dataOrig)    
    # Plot slices of Resmap volume
    plotResMapVolume(resTOTALma, minRes, maxRes)
    # Plot resolution histogram
    plotResolutionHistogram(resHisto)

    plt.show()
    

def plotOriginalVolume(volumeData):    
    fig, _ = plotVolumeSlices('Slices Through Input Volume', volumeData, np.min(volumeData), np.max(volumeData), plt.cm.gray)
    return fig

def plotResMapVolume(resmapData, minRes, maxRes):
    fig, im = plotVolumeSlices('Slices Through ResMap Results', resmapData, minRes, maxRes, plt.cm.jet)
    cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
    fig.colorbar(im, cax=cax)
    return fig
    
def plotVolumeSlices(title, volumeData, vminData, vmaxData, cmap, **kwargs):
    """ Helper function to create plots of volumes slices. 
    Params:
        title: string that will be used as title for the figure.
        volumeData: numpy array representing a volume from where to take the slices.
        cmap: color map to represent the slices.
    """
    # Get some customization parameters, by providing with default values
    titleFontSize = kwargs.get('titleFontSize', 14)
    titleColor = kwargs.get('titleColor','#104E8B')
    sliceFontSize = kwargs.get('sliceFontSize', 10)
    sliceColor = kwargs.get('sliceColor', '#104E8B')
    size = volumeData.shape[0]
    
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    f.suptitle(title, fontsize=titleFontSize, color=titleColor, fontweight='bold')
    
    def showSlice(ax, index):
        pos = int(index*size/9)
        ax.set_title('Slice ' + str(pos), fontsize=sliceFontSize, color=sliceColor)
        return ax.imshow(volumeData[pos,:,:], vmin=vminData, vmax=vmaxData, cmap=cmap, interpolation="nearest")
    
    im = showSlice(ax1, 3)
    showSlice(ax2, 4)
    showSlice(ax3, 5)
    showSlice(ax4, 6)
    
    return f, im 

def plotResolutionHistogram(histogramData, **kwargs):
    # Histogram
    f3   = plt.figure()
    f3.suptitle('Histogram of ResMap Results', fontsize=14, color='#104E8B', fontweight='bold')
    axf3 = f3.add_subplot(111)

    axf3.bar(range(len(histogramData)), histogramData.values(), align='center')
    axf3.set_xlabel('Resolution (Angstroms)')
    axf3.set_xticks(range(len(histogramData)))
    axf3.set_xticklabels(histogramData.keys())
    axf3.set_ylabel('Number of Voxels')
    
    return f3
    