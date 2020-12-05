import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
from scipy.signal import savgol_filter
from scipy import stats
from scipy import optimize
import pandas as pd
import seaborn as sns
from pyproj import Proj, transform
from numpy import sin, radians

def compute_s_coord(x,y):
    # function for computing along-centerline coordinates
    dx = np.diff(x); dy = np.diff(y)      
    ds = np.sqrt(dx**2+dy**2)
    ds = ds.astype(int)
    s = np.hstack((0,np.cumsum(ds)))
    return dx,dy,ds,s
    
def calc_R(xc,yc,x1,y1):
    """ calculate the distance of each 2D points from the center (xc, yc)"""
    return np.sqrt((x1-xc)**2 + (y1-yc)**2)

def f_2(c,x1,y1):
    """ calculate the algebraic distance between the 2D points and the mean circle centered at c=(xc, yc) """
    Ri = np.sqrt((x1-c[0])**2 + (y1-c[1])**2)
    return Ri - Ri.mean()

def analyze_cline(x,y,delta_s,smoothingf):
    # function for analyzing channel centerlines
    #
    # INPUTS:
    # x - x coordinate of centerline
    # y - y coordinate of centerline
    # delta_s - desired distance between consecutive points on centerline
    # smoothing_f - smoothing factor used by Savitzky-Golay filter (has to be an odd number)
    #
    # OUTPUTS:
    # arc_length - arc length of each meander bend (same units as x and y)
    # half_wave_length - half wave length of each bend (same units as x and y)
    # sinuosity - sinuosity of each bend (calculated using arc length / half wave length)
    # asymmetry - asymmetry of each bend: difference between upstream and downstream parts of meander
    # (relative to location of maximum curvature), normalized by arc length
    # loc_max_curv - location of maximum curvature: indices of xi and yi coordinates for these locations
    # max_curv - value of maximum curvature (for each bend)
    # x_infl - x coordinates of inflection points
    # y_infl - y coordinates of inflection points
    # xi - new, resampled x coordinates 
    # yi - new, resampled y coordinates
    # fig1 - label for map-view figure
    # fig2 - label for curvature plot
    
    dx,dy,ds,s = compute_s_coord(x,y)
    # resample centerline so that 'delta_s' is roughly constant:
    tck, u = scipy.interpolate.splprep([x,y],s=0) # parametric spline representation of curve
    unew = np.linspace(0,1,int(1+sum(ds)/delta_s)) # vector for resampling
    out = scipy.interpolate.splev(unew,tck) # resampling
    xi = out[0]
    yi = out[1]
    xi = savgol_filter(xi, 21, 3) # smoothing
    yi = savgol_filter(yi, 21, 3) # smoothing
    dx,dy,ds,si = compute_s_coord(xi,yi) # recalculate derivatives and s coordinates
    ddx = np.diff(dx); ddy = np.diff(dy) # second derivatives 
    curvature = (dx[1:]*ddy - dy[1:]*ddx) / ((dx[1:]**2 + dy[1:]**2)**1.5)
    # smoothing with the Savitzky-Golay filter:
    curvature2 = savgol_filter(curvature, smoothing_f, 3)
    xi = xi.astype(int) # convert xi and yi to integers for line 145/146
    yi = yi.astype(int)

 # Find inflection points:
    n_curv = abs(np.diff(np.sign(curvature2)))
    n_curv[plt.mlab.find(n_curv==2)] = 1
    loc_zero_curv = plt.mlab.find(n_curv)
    loc_zero_curv = loc_zero_curv +1
    n_infl = np.size(loc_zero_curv)
    # x-y coordinates of inflection points:
    x_infl = xi[loc_zero_curv]
    y_infl = yi[loc_zero_curv]

    # Find locations of maximum curvature and calculate bend asymmetry:
    max_curv = np.zeros(n_infl-1)
    loc_max_curv = np.zeros(n_infl-1, dtype=int)
    asymmetry = np.zeros(n_infl-1)
    for i in range(1, n_infl):
        if np.mean(curvature[loc_zero_curv[i-1]:loc_zero_curv[i]])>0:
            max_curv[i-1] = np.max(curvature[loc_zero_curv[i-1]:loc_zero_curv[i]])
        else:
            max_curv[i-1] = np.min(curvature[loc_zero_curv[i-1]:loc_zero_curv[i]])
        loc_max_curv[i-1] = loc_zero_curv[i-1] + plt.mlab.find(curvature[loc_zero_curv[i-1]:loc_zero_curv[i]]==max_curv[i-1])
        asymmetry[i-1] = (abs(si[loc_zero_curv[i-1]]-si[loc_max_curv[i-1]])-abs(si[loc_max_curv[i-1]]-si[loc_zero_curv[i]]))/abs(si[loc_zero_curv[i-1]]-si[loc_zero_curv[i]]);

    # Calculate half wavelengths, arc lengths, and sinuosity:
    half_wave_length = np.zeros(n_infl-1)
    arc_length = np.zeros(n_infl-1, dtype=float)
    for i in range(0,n_infl-1):
        half_wave_length[i] = np.sqrt((x_infl[i+1]-x_infl[i])**2 + (y_infl[i+1]-y_infl[i])**2)
        arc_length[i] = abs(si[loc_zero_curv[i+1]]-si[loc_zero_curv[i]])
    sinuosity = arc_length/half_wave_length
    
    fig1 = plt.figure(figsize=(12,8))
    ax = fig1.add_subplot(1,1,1)
    
    half_window = round(np.mean(arc_length)/(10.0*delta_s))
    half_window = half_window.astype(int)
    
    Rs = []
    for i in range(len(loc_max_curv)):
        x1 = xi[loc_max_curv[i]-half_window:loc_max_curv[i]+half_window+1]
        y1 = yi[loc_max_curv[i]-half_window:loc_max_curv[i]+half_window+1]

        x_m = np.mean(x1)
        y_m = np.mean(y1)
        center_estimate = [x_m, y_m]
        center_2, ier = optimize.leastsq(f_2,center_estimate,args=(x1,y1))
    
        xc_2, yc_2 = center_2
        Ri_2       = calc_R(xc_2,yc_2,x1,y1)
        R_2        = Ri_2.mean()
        Rs.append(R_2)
        
        if sinuosity[i]>1.01:
            plt.plot(x1,y1,'.b')
            circle1=plt.Circle((xc_2,yc_2),R_2,color='r',fill=False,linewidth=1)
            fig1.gca().add_artist(circle1) 
        plt.savefig("fig1.pdf", format="pdf")
    
        
    # PLOTTING:
    plt.plot(xi,yi,'k',linewidth = 0.3)
    plt.rcParams.update({'font.size': 6})
    plt.plot(xi[0], yi[0], 'ro', markersize = 0.3)
    for i in range(1, n_infl):
        c = np.array([np.remainder(i,2), 0, np.remainder(i+1,2)])
        plt.plot(xi[loc_zero_curv[i-1] : loc_zero_curv[i]+1], yi[loc_zero_curv[i-1] : loc_zero_curv[i]+1], color = c, linewidth = 0.1)
    plt.plot(x_infl, y_infl, 'ko', markersize = 1)
    plt.plot(xi[loc_max_curv], yi[loc_max_curv], 'go', markersize = 0.3)
    plt.axis('equal')
    for i in range(0, n_infl-1):
        plt.text(xi[int(round((loc_zero_curv[i] + loc_zero_curv[i+1])/2))], yi[int(round((loc_zero_curv[i] + loc_zero_curv[i+1])/2))], '%3.3g' % (sinuosity[i]))
    plt.savefig("fig2.pdf", format="pdf")
    fig2 = plt.figure(figsize=(20,20))
    
    plt.plot(si[1:-1:10], 0 * si[1:-1:10], 'k',linewidth = 0.3)
    plt.plot(si[1:-1], curvature, 'b', label = 'curvature', linewidth = 0.3)
    plt.plot(si[1:-1], curvature2, 'r',  label = 'smooth curvature', linewidth = 0.3)
    for i in range(0, n_infl-1):
        plt.text(si[loc_max_curv[i]],curvature2[loc_max_curv[i]],'%3.3g' % (sinuosity[i]))
    plt.legend(loc='upper right')
    plt.xlabel('distance along centerline (m)')
    plt.ylabel('curvature')
    
    plt.savefig("fig3.pdf", format="pdf")
    fig3 = plt.figure(figsize=(24,6))
    return [curvature, arc_length, half_wave_length, sinuosity, asymmetry, loc_max_curv, 
            max_curv, Rs, x_infl, y_infl, xi, yi, si, fig1, fig2]

df = pd.read_csv("ChSev_centreline.csv")
x = df["x"]
y = df["y"]
delta_s = 50
smoothing_f = 51
curvature, arc_length, half_wave_length, sinuosity, asymmetry, loc_max_curv, max_curv, Rs, x_infl, y_infl, xi, yi, si, fig1, fig2 = analyze_cline(x,y,delta_s,smoothing_f)
# p1 = Proj(proj='latlong', ellps='WGS84')
# p2 = Proj(proj='utm', zone='32', ellps='WGS84')
# lon, lat = transform(p2, p1, xi, yi)
# create Pandas dataframe:
Niger_bends = pd.DataFrame(np.vstack((x_infl,y_infl)).transpose())
Niger_bends.columns = ['x_infl','y_infl']
Niger_bends['sinuosity'] = np.hstack((sinuosity,np.nan))
Niger_bends['max_curv'] = np.hstack((max_curv,np.nan))
Niger_bends['half_wave_length'] = np.hstack((half_wave_length,np.nan))
Niger_bends['arc_length'] = np.hstack((arc_length,np.nan))
Niger_bends['x_max_curv'] = np.hstack((xi[loc_max_curv],np.nan))
Niger_bends['y_max_curv'] = np.hstack((yi[loc_max_curv],np.nan))
Niger_bends['asymmetry'] = np.hstack((asymmetry,np.nan))
Niger_bends['radius'] = np.hstack((Rs,np.nan))

# Exporting the dataframe to excel
Niger_bends.to_csv(r'/Users/whamitchell/Documents/python/channel_sinuosities/Niger_ChSev.csv', index = False) 
# export_excel = Niger_bends.to_excel(r'Ch7_DeltaS_50.xlsx', index = None, header=True)