# =================================================================================================================================================
#                                       Import modules

import pickle
import numpy.random as random
from numpy import linalg as LA
import numpy as np

# =================================================================================================================================================
#                                       Functions

def PickleIt(data,fileName):
    with open(fileName, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def GetPickle(fileName):
    with open(fileName, 'rb') as f:
        data = pickle.load(f)
    return data

def ActFunc(n,maxim,minim,value,off):
    return np.exp((-n/(3*(maxim-minim)))*(value-off)**2)

def NeuronFunc(x):
    return (1.+np.tanh(x-15.))*.5 # 4.=9 input

def InputFunc(x,dim,nAct,maxAcc,minAcc):
    rangeAcc=maxAcc-minAcc
    off=np.arange(minAcc+rangeAcc/(2.*nAct),maxAcc,rangeAcc/nAct)
    res=np.array([[ActFunc(nAct,maxAcc,minAcc,x[j],off[i]) for i in range(len(off))] for j in range(len(x))])
    res=res.reshape(res.size,1)
    for i in range(res.size):
        if res[i]<1e-2:
            res[i]=.0
    return res



# =================================================================================================================================================
#                                       Global Variables

# For time shifting of accelerations
ax=.1
ay=.1
az=.1

ax2=.1
ay2=.1
az2=.1

ax3=.1
ay3=.1
az3=.1

ax4=.1
ay4=.1
az4=.1

ax5=.1
ay5=.1
az5=.1

# Variables for average accelerations
ax_avg=.1
ay_avg=.1
az_avg=.1


# =================================================================================================================================================
#                                       Creating muscles
control_ids = {}

control_ids["Leg1_FLEX"] = setSingleLinearMuscle(reference_object_name = "Leg1", reference_object_application_point = [0.0,0.0, 0.6], attached_object_name = "Center", attached_object_application_point = [0.0,0.0, 1.0], maxF = 40.0)
control_ids["Leg1_EXT" ] = setSingleLinearMuscle(reference_object_name = "Leg1", reference_object_application_point = [0.0,0.0,-0.6], attached_object_name = "Center", attached_object_application_point = [0.0,0.0,-1.0], maxF = 40.0)
control_ids["Leg2_FLEX"] = setSingleLinearMuscle(reference_object_name = "Leg2", reference_object_application_point = [0.0,0.0, 0.6], attached_object_name = "Center", attached_object_application_point = [0.0,0.0, 1.0], maxF = 40.0)
control_ids["Leg2_EXT"]  = setSingleLinearMuscle(reference_object_name = "Leg2", reference_object_application_point = [0.0,0.0,-0.6], attached_object_name = "Center", attached_object_application_point = [0.0,0.0,-1.0], maxF = 40.0)
control_ids["Leg3_FLEX"] = setSingleLinearMuscle(reference_object_name = "Leg3", reference_object_application_point = [0.0,0.0, 0.6], attached_object_name = "Center", attached_object_application_point = [0.0,0.0, 1.0], maxF = 40.0)
control_ids["Leg3_EXT"]  = setSingleLinearMuscle(reference_object_name = "Leg3", reference_object_application_point = [0.0,0.0,-0.6], attached_object_name = "Center", attached_object_application_point = [0.0,0.0,-1.0], maxF = 40.0)
control_ids["Leg4_FLEX"] = setSingleLinearMuscle(reference_object_name = "Leg4", reference_object_application_point = [0.0,0.0, 0.6], attached_object_name = "Center", attached_object_application_point = [0.0,0.0, 1.0], maxF = 40.0)
control_ids["Leg4_EXT"]  = setSingleLinearMuscle(reference_object_name = "Leg4", reference_object_application_point = [0.0,0.0,-0.6], attached_object_name = "Center", attached_object_application_point = [0.0,0.0,-1.0], maxF = 40.0)



# =================================================================================================================================================
#                                       Network creation
np.random.seed(np.random.randint(0,10000))
dim=8 # Dimension of input
nAct=10 # Projection of input values onto the input neurons of reservoir
nInp=nAct*dim # Total input dimension
nRes=100 # Number of elements(neurons) in reservoir
nOut=4 # Number of output values

wInp=np.random.rand(nRes,nInp) # Weights from Input neurons to Reservoir Neurons

wRes=2.*np.random.rand(nRes*nRes,1) # Weights of self connection inside reservoir
indInh=np.random.randint(0,wRes.size,size=wRes.size*.1) # Number of inhibitory connections
wRes[indInh]*=-1.
wRes=wRes.reshape(nRes,nRes)
wRes*=1.25/max(abs((LA.eigvals(wRes))))

wOut=np.random.rand(nOut,nInp+nRes) # Weigths from readouts and inputs to the outputs

out=np.random.rand(nOut,1)

x=np.random.rand(nRes,1) #States of reservoir
u=np.random.rand(nInp,1) #Inputs to reservoir

## Record the accelerations, readouts, inputs and outputs for 100 time steps
axT=np.zeros((100,1))
RecT=np.zeros((100,(nInp+nRes+nOut)))
intercept=np.zeros((nOut,1)) # Intercept of ridge regression of readouts on outputs

Record_or_Test=0    #0: Record (Gather Data), 1: Test , 2: Use test parameters and record the data also 

if Record_or_Test==0:
    PickleIt([wInp,wRes],'paramStatic') # Record the weights to use in test case
    Record=np.zeros((1,(nInp+nRes+nOut))) # Initialize record variable
elif Record_or_Test==1: # Test case
    aa=GetPickle('paramStatic1')
    wInp=aa[0]
    wRes=aa[1]
    wOut=GetPickle('wout') # Get regressed output weights
elif Record_or_Test==2: # Test case
    aa=GetPickle('paramStatic1')
    wInp=aa[0]
    wRes=aa[1]
    wOut=GetPickle('wout')
    intercept=GetPickle('intercept')
    intercept=np.zeros((nOut,1)) # Not to use intercept for now
    #intercept=intercept.reshape(nOut,1)
    PickleIt([wInp,wRes],'paramStatic')
    Record=np.zeros((1,(nInp+nRes+nOut)))



# =================================================================================================================================================
#                                       Evolve function
def evolve():
    ## Global Variables
    global x,wInp,u,wRes,wOut,nInp,nAct,Inc,nOut,out,ax,ay,az,ax2,ay2,az2,ax3,ay3,az3,ax4,ay4,az4
    global dim,ax5,ay5,az5,Record, Record_or_Test
    global ax_avg,ay_avg,az_avg,axT,RecT
    
    print("Step:", i_bl, "  Time:{0:.2f}".format(t_bl),'   Acc:{0:8.2f}  {1:8.2f}  {2:8.2f}'.format(ax,ay,az),\
          'Acc:{0:8.2f} {1:8.2f} {2:8.2f}'.format(ax_avg,ay_avg,az_avg))
    # ------------------------------------- Visual ------------------------------------------------------------------------------------------------
    #visual_array     = getVisual(camera_name = "Meye", max_dimensions = [256,256])
    #scipy.misc.imsave("test_"+('%05d' % (i_bl+1))+".png", visual_array)
    # ------------------------------------- Olfactory ---------------------------------------------------------------------------------------------
    olfactory_array  = getOlfactory(olfactory_object_name = "Center", receptor_names = ["smell1", "plastic1"])
    # ------------------------------------- Taste -------------------------------------------------------------------------------------------------
    taste_array      = getTaste(    taste_object_name =     "Center", receptor_names = ["smell1", "plastic1"], distance_to_object = 1.0)
    # ------------------------------------- Vestibular --------------------------------------------------------------------------------------------
    vestibular_array = getVestibular(vestibular_object_name = "Center")
    # ------------------------------------- Sensory -----------------------------------------------------------------------------------------------
    # ------------------------------------- Proprioception ----------------------------------------------------------------------------------------
    sF1 = getMuscleSpindle(control_id = control_ids["Leg1_FLEX"])[0:2]
    sE1  = getMuscleSpindle(control_id = control_ids["Leg1_EXT"])[0:2]
    sF2 = getMuscleSpindle(control_id = control_ids["Leg2_FLEX"])[0:2]
    sE2  = getMuscleSpindle(control_id = control_ids["Leg2_EXT"])[0:2]
    sF3 = getMuscleSpindle(control_id = control_ids["Leg3_FLEX"])[0:2]
    sE3  = getMuscleSpindle(control_id = control_ids["Leg3_EXT"])[0:2]
    sF4 = getMuscleSpindle(control_id = control_ids["Leg4_FLEX"])[0:2]
    sE4  = getMuscleSpindle(control_id = control_ids["Leg4_EXT"])[0:2]

    #print( sF1)
    
    # ------------------------------------- Neural Simulation -------------------------------------------------------------------------------------
    
    
    
    
    
    ax=vestibular_array[3] # Read accelerations
    ay=vestibular_array[4]
    az=vestibular_array[5]

    if Record_or_Test==0: # Record Case
        axT=np.vstack((ax,axT[:-1])) # Record and shift accelerations on x-axis
        RecT=np.vstack((np.vstack((x,u,out)).T,RecT[:-1,:])) # Record and shift readouts and output
        
        ax_avg=np.sum(axT)/100. # Average acceleration on 100 steps on x-axis
        #ax_avg=(ax_avg*(i_bl)+ax)/(i_bl+1)
        ay_avg=(ay_avg*(i_bl)+ay)/(i_bl+1) # Velocity on y-axis ?
        az_avg=(az_avg*(i_bl)+az)/(i_bl+1)
        if ax_avg>5. and ay_avg<1. and ay_avg>-1.: # Record the outputs if acceleration is positive
            z=np.vstack((x,u))
            Record=np.vstack((Record,RecT))
            #Record=np.vstack((Record,np.vstack((z,out)).T))
            #print(Record.shape,np.vstack((z,out)).T.shape)    
        if np.mod(i_bl,1000)==0: # Save once in 1000 steps
            PickleIt(Record,'paramWalking')
    elif Record_or_Test==2:
        axT=np.vstack((ax,axT[:-1]))
        ax_avg=np.sum(axT)/100.
        ay_avg=(ay_avg*(i_bl)+ay)/(i_bl+1)
        az_avg=(az_avg*(i_bl)+az)/(i_bl+1)
        #ax_avg=scn.objects["Center"].localLinearVelocity[0]
        #ay_avg=scn.objects["Center"].localLinearVelocity[1]
        #az_avg=scn.objects["Center"].localLinearVelocity[2]
        if ax_avg>20. and ay_avg<1. and ay_avg>-1.: # Record the outputs if acceleration is positive
            z=np.vstack((x,u))
            Record=np.vstack((Record,np.vstack((z,out)).T))
            #print(Record.shape,np.vstack((z,out)).T.shape)    
        if np.mod(i_bl,1000)==0: # Save once in 1000 steps
            PickleIt(Record,'paramWalking')
    else:
        axT=np.vstack((ax,axT[:-1]))
        ax_avg=np.sum(axT)/100.
        ay_avg=(ay_avg*(i_bl)+ay)/(i_bl+1)
        az_avg=(az_avg*(i_bl)+ax)/(i_bl+1)

    #x=(1-a)*x+0.3*NeuronFunc(wInp.dot(np.vstack((u,out)))+wRes.dot(x)) # States get output as input
    x=(1-(a))*x+0.3*NeuronFunc(wInp.dot(u)+wRes.dot(x)) # States dont have outputs as inputs
    


    #input_array=[ax,ay,az,ax2,ay2,az2,ax3,ay3,az3]
    #input_array1=[ax,ay,az]
    input_array2=[sF1[0],sF1[1],sF2[0],sF2[1],sF3[0],sF3[1],sF4[0],sF4[1]] # Inputs are only muscle positions

    #input_array1=InputFunc(input_array1,len(input_array1),nAct,100.,-100.)
    input_array2=InputFunc(input_array2,len(input_array2),nAct,2.,-2.) # Convert muscle positions to inputs of the ESN
    
    u=input_array2

    u=u.reshape(nInp,1) # Reshape inputs (n,1)
    x=x.reshape(nRes,1)
    z=np.vstack((x,u))

    if Record_or_Test==1: # Test case with using Regressed wOut
        out=wOut.dot(z) # Calculate outputs with using wOut
        #out=out/np.amax(out) # This scale needed not to diverge ???
        out.reshape(nOut,1)
    elif Record_or_Test==0: # Record case with random outputs 
        out=np.random.rand(nOut,1)      
    elif Record_or_Test==2:
        out=wOut.dot(z)+intercept # Calculate outputs with using wOut
        #out=out/np.amax(out) # This scale needed not to diverge ???
        out.reshape(nOut,1)

    # ------------------------------------- Muscle Activation -------------------------------------------------------------------------------------
    freq=1.
    act_tmp1      = 0.5+0.5*np.sin(freq*t_bl+out[0])
    anti_act_tmp1 = 1.0 - act_tmp1
    act_tmp2      = 0.5+0.5*np.sin(freq*t_bl+out[1])
    anti_act_tmp2 = 1.0 - act_tmp2
    act_tmp3      = 0.5+0.5*np.sin(freq*t_bl+out[2])
    anti_act_tmp3 = 1.0 - act_tmp3
    act_tmp4      = 0.5+0.5*np.sin(freq*t_bl+out[3])
    anti_act_tmp4 = 1.0 - act_tmp4


    # Muscles
    controlActivity(control_id = control_ids["Leg1_FLEX"], control_activity = act_tmp1)
    controlActivity(control_id = control_ids["Leg1_EXT"] , control_activity = anti_act_tmp1)
    controlActivity(control_id = control_ids["Leg2_FLEX"], control_activity = act_tmp2)
    controlActivity(control_id = control_ids["Leg2_EXT"] , control_activity = anti_act_tmp2)
    controlActivity(control_id = control_ids["Leg3_FLEX"], control_activity = act_tmp3)
    controlActivity(control_id = control_ids["Leg3_EXT"] , control_activity = anti_act_tmp3)
    controlActivity(control_id = control_ids["Leg4_FLEX"], control_activity = act_tmp4)
    controlActivity(control_id = control_ids["Leg4_EXT"] , control_activity = anti_act_tmp4)

    #print(x)
## Check Accelerations and Record ###



#####################################
    



#bge.logic.endGame()
#bge.logic.restartGame()
#scn.reset()
