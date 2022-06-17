## -*- coding: utf-8 -*-
## INVERSION.PY - Inverts data from Airmodus A09 PSM data in to size distributions based on Lehtipalo et al, 2014
## Use: Input file path in filename variable and adjust dataDate to fit date of interest.
## Dependencies:
##      Numpy, matplotlib, scipy
## Categories:
##      Aerosol nucleation
## Author:
##      Lee Tiszenkel, UAH
## Date created:
##      5/21/2021
## Doop

import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema, find_peaks
from statistics import mean
import matplotlib.dates as mdates
import matplotlib.ticker as ticker

dataDate="20220527"
dataDay=dataDate[-4:]
#filename = "C:/Users/kingg/Documents/data/SPRING-2021-DATA/3521/A0911010007_ZERO_DAT{}.dat".format(dataDate)
#filename = "C:/Users/kingg/Documents/data/SPRING-2022/520/UAH_A10_{}.dat".format(dataDate)
filename = "UAH_A10_{}.dat".format(dataDate)

calibPlot = True
plotting = True

pi = 3.1415927

#Fits/smoothing functions
def moving_average(x, w):
    return np.convolve(x, np.ones(w), "same") / w
def power_law(x, a, b):
    return a*np.power(x, b)
def exponential(x, a, b, c, d):
    return (a*np.exp(b*x))+(c*np.exp(d*x))
def visc(t):
    return (174+0.433*(t-273))*1e-7
def rlambda(t, press):
    dm = 3.7e-10
    avog = 6.022e23
    return 8.3143*t/((2**(1/2))*avog*press*pi*dm*dm)
def cunn(dp, t, press):
    return 1.0+rlambda(t,press)/dp*(2.514+0.8*np.exp(-0.55*dp/rlambda(t,press)))
def reynolds(flow, diameter, t, press):
    density = 1.29*(273/t)*(press/101325)
    pipearea = pi/4*diameter**2
    velocity = flow/pipearea
    return density*velocity*pipearea/visc(t)
def diffusion(dpp, temp, press):
    K = 1.38e-23
    return (K*temp*cunn(dpp,temp,press))/(3*pi*visc(temp)*dpp)
def ltubefl(dpp, plength, pflow, temp, press):
    rmuu = pi*diffusion(dpp, temp, press)*(plength/pflow)
    i = 0
    res = []
    while i < len(dpp):
        if rmuu[i] < 0.02:
            res.append(1-2.56*rmuu[i]**(2/3)+1.2*rmuu[i]+0.177*rmuu[i]**(4/3))
        else:
            res.append(0.819*np.exp(-3.657*rmuu[i])+0.097*np.exp(-22.3*rmuu[i])+0.032*np.exp(-57*rmuu[i]))
        i += 1
    return res

#Retrieving data
print("Retrieving data...")
time = np.genfromtxt(filename, delimiter=",", skip_header=1, usecols=0, dtype="str")
timeDT = []
for x in time:
    timeDT.append(dt.datetime.strptime(x, "%d.%m.%Y %H:%M:%S"))
conc = np.genfromtxt(filename, delimiter=",", skip_header=1, usecols=1)
satflow = np.genfromtxt(filename, delimiter=",", skip_header=1, usecols=3)

timeUS = []
for x in timeDT:
    timeUS.append(dt.datetime.strftime(x, "%m/%d/%Y %H:%M:%S"))
concSmooth = moving_average(conc,2)

flowmax = round(100*max(satflow))/100
flowmin = round(100*min(satflow))/100

#Retrieving calibration data
calibFilename = "C:/Users/kingg/Documents/data/SPRING-2022/520/A10calib.txt"
print("Retrieving calibration data from " + calibFilename)
calibSat = np.genfromtxt(calibFilename, usecols=0)
calibDia = np.genfromtxt(calibFilename, usecols=1)
calibDet = np.genfromtxt(calibFilename, usecols=2)

#Size bin cutoffs
#dia = [3.1,2.8,2.5,2.0,1.7,1.5,1.35,1.25]
dia = calibDia.tolist()
diaCut = []
i = 0
while i < len(dia)-1:
    diaCut.append((dia[i]+dia[i+1])/2)
    i += 1
print("Done.")

        
#Plotting calibration data, curve/parameter fitting
print("Parameterizing calibration...")
pars1, cov1 = curve_fit(f=exponential, xdata=calibDia, ydata=calibSat, p0=[0, 0, 0, 0], bounds=(-np.inf, np.inf))
pars2, cov2 = curve_fit(f=exponential, xdata=calibDia, ydata=calibDet, p0=[0, 0, 0, 0], bounds=(-np.inf, np.inf))
pars3, cov3 = curve_fit(f=exponential, xdata=calibSat, ydata=calibDia, p0=[0, 0, 0, 0], bounds=(-np.inf, np.inf))
    
#stdevs = np.sqrt(np.diag(cov1))
#stdevs = np.sqrt(np.diag(cov2))

y1 = []
y2 = []
y3 = []
for x in diaCut:
    y1.append(exponential(x, *pars1))
    y2.append(exponential(x, *pars2))
for x in satflow:
    y3.append(exponential(x, *pars3))

apu = np.array(range(0, len(diaCut)+1, 1))
a = (flowmax/flowmin)**(1/(len(diaCut)))
P0 = flowmax/a**len(diaCut)
satflowAM = (P0*(a**apu))
print("Done.")
print("Determining number of scans...")
##Enumerating scans
satflowSmooth = moving_average(satflow, 2)
#k11 = argrelextrema(satflowSmooth, np.less, order=3)
k11 = find_peaks(satflow, plateau_size=5)[0]
k1 = np.append(k11, len(satflowSmooth))
nscan = []
k2 = np.amin(k1)
i=0
j=1

for g in k1:
    while i < int(g):
        nscan.append(j)
        i+=1
    j += 1
numScans = str(max(nscan))
print(numScans + " scans found.")
print("Averaging bins for each scan...")
##Averaging bins for each scan
lr = len(satflowAM)
timenew_avg = []
meanflow = []
conc1b = []
diameter = []
i = 2
ii = 1
iii = 1

npNscan = np.array(nscan)
while iii < lr:
    meanflow.append((satflowAM[iii-1]+satflowAM[iii])/2)
    diameter.append((dia[iii-1]+dia[iii])/2)
    iii += 1

while i < np.max(npNscan):
    tempConc = []
    k = np.where(npNscan == i)
    timenew_avg.append(timeUS[int(np.mean(k[0]))])
    concMx = []
    satMx = []
    for b in k[0]:
        concMx.append(conc[b])
        satMx.append(satflow[b])
    npconcMx = np.array(concMx)
    npsatMx = np.array(satMx)
    for c in meanflow:
        if c == min(meanflow):
            k1 = np.where(npsatMx < c)
            k2 = np.where(npsatMx > satflowAM[0])
        elif c == max(meanflow):
            k1 = np.where(npsatMx > meanflow[-1])
            k2 = np.where(npsatMx < satflowAM[-1])
        else:
            k1 = np.where(npsatMx > meanflow[(meanflow.index(c))-1])
            k2 = np.where(npsatMx < c)
        k3 = np.intersect1d(k1, k2)
        try:
            avgConc = mean(concMx[x] for x in k3)
            tempConc.append(avgConc)
        except:
            tempConc.append(0)
    conc1b.append([i, tempConc])
    i += 1
print("Done.")
print("Calculating detection efficiency...")
DETEFF = []
for d in diameter:
    DETEFF.append(exponential(d, *pars2))
print("Done.")
#M = []

#for g in conc1b:                            
#    cMax = np.nanmax(g[1])                  
#    cf = np.polyfit(meanflow,g[1]/cMax,1)   
#    M.append([i, cf[1]])

dconc = []
for g in conc1b:
    tempdConc = []
    i = 0
    while i<len(g[1])-1:
        tempdConc.append((g[1][i+1]-g[1][i])/DETEFF[i])
        i += 1
    dconc.append([g[0], tempdConc])

##Loss correction

#Length of tube (cm)
inletl = 19.05
#Radius of tube (cm)
inletr = 0.635
#Inlet flow rate (LPM)
inletfr = 2.5
#temperature
temp = 298
#pressure
press = 101325

print("Calculating diffusional losses...")
inletl = inletl * 1e-2
inletr = inletr * 1e-2

d = np.array(diaCut) * 1e-9
Q = inletfr*0.001/60

rooi = 1.205
roop = 1000
vis = visc(temp)
A = pi*(inletr**2)
U = Q/A
Re = reynolds(Q,inletr*2,temp,press)

D = diffusion(d, temp, press)

Vd = 0
Pt = 1
Pl = ltubefl(d, inletl, Q, temp, press)
npPl = np.array(Pl)
print("Done.")
    
print("Correcting size distribution...")
concCorr = []
for g in conc1b:   
    conx = np.array(g[1])
    conxCorr = conx/npPl
    concCorr.append(conxCorr)

dconc = []
dconcNans = []
for g in concCorr:
    tempdConc = []
    tempdConcNans = []
    i = 0
    while i<len(g)-1:
        value = (g[i+1]-g[i])/DETEFF[i]
        if value < 0:
            tempdConc.append(0)
            tempdConcNans.append(float('nan'))
        else:
            tempdConc.append(value)
            tempdConcNans.append(value)
        #tempdConc.append((g[i+1]-g[i])/DETEFF[i])
        i += 1
    dconc.append(tempdConc)
    dconcNans.append(tempdConcNans)
print("Done.")
print("Calcuating dN/dLogDp...")
#Calculating dLogDp
i=0
dlogdp = []
while i < len(diaCut)-1:
    dlogdp.append(np.log10(diaCut[i+1]*1e-9)-np.log10(diaCut[i]*1e-9))
    i += 1

#conc4 is dN/dlogDp
NPdlogdp = np.array(dlogdp)
conc4 = []
conc4Nans = []

for g in dconc:
    conc4.append(g/NPdlogdp)
for g in dconcNans:
    conc4Nans.append(g/NPdlogdp)

concTime = zip(timenew_avg, concCorr)
dconcTime = zip(timenew_avg, conc4)

dtTimenew = []
for x in timenew_avg:
    dtTimenew.append(dt.datetime.strptime(x, "%m/%d/%Y %H:%M:%S"))
figbins = []
i=0
while i < len(diaCut)-1:
    figbins.append((diaCut[i]+diaCut[i+1])/2)
    i += 1
print("Done.")
print("Creating output files...")
i=0
with open('{}-PSMconcDN.txt'.format(dataDate), 'w') as fp:
    fp.write("{}\n".format(",".join([str(y) for y in figbins])))
    while i<len(timenew_avg):
        fp.write("{},{}\n".format(timenew_avg[i], ",".join([str(y) for y in conc4[i]])))
        i += 1

i=0
with open('{}-PSMconc.txt'.format(dataDate), 'w') as fp:
    while i<len(timenew_avg):
        fp.write("{},{}\n".format(timenew_avg[i], ",".join([str(y) for y in concCorr[i]])))
        i += 1
print("Done.")
if plotting == True:
    print("Plotting...")
    if calibPlot == True:
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(3,1,1)
        ax2 = fig.add_subplot(3,1,2)
        ax3 = fig.add_subplot(3,1,3)
        ax.scatter(calibDia, calibSat)
        ax2.scatter(calibDia, calibDet)
        smoothX = np.linspace(min(calibDia),max(calibDia), 100)
        ax.plot(smoothX, exponential(smoothX, *pars1), linewidth=2, linestyle='--', color='black')
        ax2.plot(smoothX, exponential(smoothX, *pars2), linewidth=2, linestyle='--', color='black')
        ax3.scatter(diaCut, Pl)

    fig2 = plt.figure(figsize=(15,5))
    ax4 = fig2.add_subplot(1,1,1)
    fig3 = plt.figure(figsize=(15,5))
    ax5 = fig3.add_subplot(1,1,1)
    fig4 = plt.figure(figsize=(15,5))
    ax6 = fig4.add_subplot(1,1,1)

    bananaPlot = ax4.contourf(dtTimenew,figbins,np.array(conc4).T+0.1, np.arange(0.1, 300000, 100), cmap='jet', extend='max')
    myFmt = mdates.DateFormatter("%m/%d/%Y") #Format date
    myFmt2 = mdates.DateFormatter("%H:%M:%S") #Format time
    ax4.xaxis.set_major_formatter(myFmt2)
    ax4.xaxis.set_major_locator(mdates.SecondLocator(interval=int(((max(dtTimenew)-min(dtTimenew)).total_seconds())/5))) #6 marks on x axis
    ax4.xaxis.set_minor_formatter(myFmt)
    ax4.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    ax4.xaxis.set_tick_params(which='minor', pad=15) #Keeps major/minor axis from overlapping
    ax4.set_yscale('log')
    ax4.yaxis.set_major_locator(ticker.MultipleLocator(0.1))

    #Size distribution in bins
    ax5.plot(dtTimenew, [y[0] for y in concCorr], label=">2.95 nm")
    ax5.plot(dtTimenew, [y[1] for y in concCorr], label=">2.65 nm")
    ax5.plot(dtTimenew, [y[2] for y in concCorr], label=">2.25 nm")
    ax5.plot(dtTimenew, [y[3] for y in concCorr], label=">1.85 nm")
    ax5.plot(dtTimenew, [y[4] for y in concCorr], label=">1.6 nm")
    ax5.plot(dtTimenew, [y[5] for y in concCorr], label=">1.5 nm")
    ax5.plot(dtTimenew, [y[6] for y in concCorr], label=">1.3 nm")
    ax5.xaxis.set_major_formatter(myFmt2)
    ax5.xaxis.set_major_locator(mdates.SecondLocator(interval=int(((max(dtTimenew)-min(dtTimenew)).total_seconds())/5))) #6 marks on x axis
    ax5.xaxis.set_minor_formatter(myFmt)
    ax5.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    ax5.xaxis.set_tick_params(which='minor', pad=15) #Keeps major/minor axis from overlapping
    ax5.set_yscale('log')
    ax5.legend()

    ax6.plot(dtTimenew, [y[0] for y in conc4Nans], label="2.65-2.95 nm")
    ax6.plot(dtTimenew, [y[1] for y in conc4Nans], label="2.25-2.65 nm")
    ax6.plot(dtTimenew, [y[2] for y in conc4Nans], label="1.85-2.25 nm")
    ax6.plot(dtTimenew, [y[3] for y in conc4Nans], label="1.6-1.85 nm")
    ax6.plot(dtTimenew, [y[4] for y in conc4Nans], label="1.5-1.6 nm")
    ax6.plot(dtTimenew, [y[5] for y in conc4Nans], label="1.3-1.5 nm")
    #ax6.plot(dtTimenew, [y[6] for y in concCorr], label=">1.3")
    ax6.xaxis.set_major_formatter(myFmt2)
    ax6.xaxis.set_major_locator(mdates.SecondLocator(interval=int(((max(dtTimenew)-min(dtTimenew)).total_seconds())/5))) #6 marks on x axis
    ax6.xaxis.set_minor_formatter(myFmt)
    ax6.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    ax6.xaxis.set_tick_params(which='minor', pad=15) #Keeps major/minor axis from overlapping
    ax6.set_yscale('log')
    ax6.legend()

    #Colorbar Setup - putting neater ticks on, label
    cb = fig2.colorbar(bananaPlot, ax = ax4)
    cb.set_label('dN/dlogDp')

    #Axis labels
    ax4.set_xlabel('Time (UTC)') 
    ax4.set_ylabel('Dp (nm)')
    fig2.savefig("{}-bananaplot.png".format(dataDate))
    fig3.savefig("{}-conc.png".format(dataDate))
    fig4.savefig("{}-dN.png".format(dataDate))
    #plt.show()