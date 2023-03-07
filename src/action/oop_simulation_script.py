import random
import numpy as np
import pandas as pd
from numba import njit, prange
from gym import Env 
from gym.spaces import Box 

@njit
def column_generate(time_slot,case='all'):
    columns=[]
    if case=='all':
        for i in range(Number_of_user):
            columns.append([j for j in range(np.sum(time_slot[i]<Departure_time[i])-tao,\
                                                      np.sum(time_slot[i]<Departure_time[i])+tao+1)])
    elif case=='chosen':
        for i in range(Number_of_user):
            columns.append([np.sum(time_slot[i]<Departure_time[i])])
    return columns

@njit
def V1(x):
    return np.square(1-x/4500)*9.78

@njit
def fic_tt(user, time_point, all_time_matrix, new_timelist, trip_len, Accumulation,Number_of_user):
    
    # get the fictional departure time for the fictional traveler
    star_time=all_time_matrix[user,time_point]
    
    # get the list of events happening after this given departure time point to simulate the expected states
    known_list=new_timelist[new_timelist>star_time]
    
    if len(known_list)==0: # if this fictional departure happens after all real travelers
        # exit the network assuming free flow speed (9.78)
        texp=trip_len[user]/9.78/60
    elif len(known_list)==Number_of_user*2: # this fictional departure happens before all real travelers enter the network
        # compute the left trip length till the first real traveler enter the network
        texp=0
        count=0
        left_len=trip_len[user]-9.78*60*(known_list[0]-star_time)
        
        if left_len<0: # if this fictional traveler end his trip before the first real traveler enter the network
            # exit the network assuming free flow speed (9.78)
            texp=trip_len[user]/9.78/60
            
        else: # compute travel speed in each time interval between two consecutive events
            V_list=np.array([V1(x) for x in Accumulation[Number_of_user*2-len(known_list):-1]])
            len_piece=np.diff(known_list)*V_list*60 # trip length traveled in each time interval between two consecutive events
            cum_len=np.cumsum(len_piece)
            count=np.sum(cum_len<left_len)
            texp=known_list[count+1]-star_time+(left_len-cum_len[count])/V1(Accumulation[count])/60
    else: # it means this fictional departure happens after some real travelers have entered the network
        texp=0
        count=0
        # compute the left trip length till the next closest event occurs (either a departure or arrival)
        left_len=trip_len[user]-V1(Accumulation[Number_of_user*2-len(known_list)-1])*(known_list[0]-star_time)*60
        if left_len<0: # if this fictional traveler end his trip before the next real event occurs
            texp=trip_len[user]/V1(Accumulation[Number_of_user*2-len(known_list)-1])/60
        else:
            # travel speed in each time interval between two consecutive events
            V_list=np.array([V1(x) for x in Accumulation[Number_of_user*2-len(known_list):-1]])
            
            # trip length traveled in each time interval between two consecutive events
            len_piece=np.diff(known_list)*V_list*60
            cum_len=np.cumsum(len_piece)
            count=np.sum(cum_len<left_len)
            if count==0:
                texp=known_list[count]-star_time+(left_len-(known_list[count]-star_time)*V1(1))/9.78/60
            elif count==len(cum_len): # this fictional traveler's is not finished even after all real travelers finish their trips
                texp=known_list[count]-star_time+(left_len-cum_len[count-1])/9.78/60
            else: # this fictional traveler finishes the trip before all real travelers finish their trips
                texp=known_list[count+1]-star_time+(left_len-cum_len[count])/V1(Accumulation[Number_of_user*2-len(known_list)+count])/60
    return texp

@njit(parallel=True)
def T_est(all_time_matrix, new_timelist, trip_len, Accumulation, Number_of_user, tao):
    T_estimate_array=np.zeros((Number_of_user,2*tao+1))
    for i in prange(Number_of_user):
        for j in prange(2*tao+1):
            T_estimate_array[i,j]=fic_tt(i, j, all_time_matrix, new_timelist, trip_len, Accumulation,Number_of_user)
    return T_estimate_array

class Simulation():
    # setting up simulation with initial parameters
    def __init__(self, params):
        np.random.seed(seed=59) # tried 2
        # setting values from parameter input
        self.omega = params['omega'] # Learning rate
        self.theta = params['theta'] #scale parameter
        self.tao = params['tao'] # number of time intervals
        self.Number_of_user = params['Number_of_user'] # number of users
        
        self.all_time_slot = pd.DataFrame()
        self.day = 0
        
        self.window_c_perceived = pd.DataFrame(columns=('t'+str(i) for i in range(2*self.tao+1)))
        self.c_perceived = pd.DataFrame(columns=('t'+str(i) for i in range(2*self.tao+1)))
        
        # departure time for each user
        self.Departure_time = np.random.normal(80,18,self.Number_of_user)
        for i in range(len(self.Departure_time)):
            if self.Departure_time[i]<20 or self.Departure_time[i]>150:
                self.Departure_time[i] = np.random.normal(80,18,1)

        # trip length for each user
        self.trip_len = np.array(np.zeros(self.Number_of_user))
        for i in range(self.Number_of_user):
            self.trip_len[i] = 4600+np.random.normal(0,(0.02*4600)**2)
            while self.trip_len[i]<20:
                self.trip_len[i] = 4600+np.random.normal(0,(0.02*4600)**2)
        
        self.Wished_Arrival_time = self.Departure_time+self.trip_len/9.78/60
        
        for i in range(2*self.tao+1):
            self.all_time_slot['t'+str(i)] = self.Departure_time-self.tao+i
        self.all_time_matrix = np.array(self.all_time_slot)

        self.E = np.array(np.zeros(self.Number_of_user)) # schedule delay early penalty
        self.L = np.array(np.zeros(self.Number_of_user)) # schedule delay late penalty
        
        self.E=np.random.lognormal(-1.9,0.2,self.Number_of_user)*4
        self.L=self.E*np.exp(1)
        self.alpha=self.E*np.exp(0.5)

        self.util_rand = np.random.gumbel(-0.57721 / self.theta, 1.0 / self.theta, (self.Number_of_user,2*self.tao+1))
        self.ur = pd.DataFrame(self.util_rand,columns=('t'+str(i) for i in range(2*self.tao+1)))

        self.Eachday_dep=pd.DataFrame()
        self.Eachday_dep['d0']=self.Departure_time
        
        self.Dep_time_set=pd.DataFrame()
        for i in range(2*self.tao+1):
            self.Dep_time_set['t'+str(i)]=self.Departure_time-self.tao+i
        
        self.Departure_time = np.load("./base_deptime.npy", allow_pickle=True)

        self.Acc_df=pd.DataFrame() # record the accumulation on each day
        self.time_label=pd.DataFrame() # time points of the events on each day

        self.cs_list = []
        self.cost_list = []
        self.ttcs_list = []
        self.utility = {}

        self.capacity = 4500

    # running a single day of simulation    
    def step(self, action):

        vehicle_information, time_list, Accumulation, Speed = self.within_day_process()
        
        self.Acc_df['d'+str(self.day)]=Accumulation
        self.time_label['d'+str(self.day)]=time_list
        vehicle_information['origin_tl']=self.trip_len
        
        new_timelist = time_list

        T_estimate = T_est(self.all_time_matrix, new_timelist, self.trip_len, Accumulation, self.Number_of_user, self.tao)
        T_estimated = pd.DataFrame(T_estimate,columns=('t'+str(i) for i in range(2*self.tao+1)))

        T_estimated_diff = pd.DataFrame()
        for j in range(2*self.tao+1):
            T_estimated_diff['t'+str(j)] = T_estimated['t'+str(j)]+self.all_time_slot['t'+str(j)]-self.Wished_Arrival_time

        T_diff = np.array(T_estimated_diff)
        SD = self.schedule_delay(T_diff)

        c_estimated=pd.DataFrame(columns=('t'+str(i) for i in range(2*self.tao+1)))
        c_cs=pd.DataFrame(columns=('t'+str(i) for i in range(2*self.tao+1)))
        ttcs=pd.DataFrame(columns=('t'+str(i) for i in range(2*self.tao+1)))

        for j in range(2*self.tao+1):
            c_estimated['t'+str(j)]=self.alpha*T_estimated['t'+str(j)]+SD[:,j]+\
                self.custgauss(self.all_time_slot['t'+str(j)],action['mu'],action['sigma'],action['A'])*np.array(vehicle_information['origin_tl'])*0.0002

            c_cs['t'+str(j)]=self.alpha*T_estimated['t'+str(j)]+SD[:,j] # consumer surplus

            ttcs['t'+str(j)]=self.alpha*T_estimated['t'+str(j)] # travel time cost
        
        if self.day==0:
            self.c_perceived=c_cs
        else:
            self.c_perceived=self.omega*self.c_perceived+(1-self.omega)*c_cs
        self.window_c_perceived = self.c_perceived
        
        utility_exp=-self.window_c_perceived+self.util_rand
        
        toll_paid = 0
        for j in range(2*self.tao+1):
            utility_exp['t'+str(j)] = utility_exp['t'+str(j)]-self.custgauss(self.all_time_slot['t'+str(j)],action['mu'],action['sigma'],action['A'])*np.array(vehicle_information['origin_tl'])*0.0002
            toll_paid += self.custgauss(self.all_time_slot['t'+str(j)],action['mu'],action['sigma'],action['A'])*np.array(vehicle_information['origin_tl'])*0.0002

        columns1 = []
        for i in range(self.Number_of_user):
            columns1.append(['t'+str(np.sum(self.all_time_slot.iloc[i].values<self.Departure_time[i]))])
        window_cs = self.rearrange(df=c_cs,cols=columns1)
        window_c_exp = self.rearrange(df=c_estimated,cols=columns1)
        window_ttcs = self.rearrange(df=ttcs,cols=columns1)
        
        self.cost_list.append(window_c_exp.sum())
        self.cs_list.append(window_cs.sum())
        self.ttcs_list.append(window_ttcs.sum())
        
        self.Departure_time=np.diag(self.Dep_time_set[utility_exp.idxmax(axis=1)])
        self.utility['d'+str(self.day)]=np.diag(self.ur[utility_exp.idxmax(axis=1)])
    
        self.Eachday_dep['d'+str(self.day+1)]=self.Departure_time
        
        # state variables
        day_idx = np.full(15, np.float32(self.day))
        
        bins = np.histogram(vehicle_information['t_dep'], bins=15, range=(-50,175))[1]
        inds = np.digitize(vehicle_information['t_dep'], bins)

        user_info = np.zeros([3,15])
        for idx, user in enumerate(inds):
            user_info[0,user]+=1 # total user departures in each 15 minute time window
            user_info[1,user]+=toll_paid[idx]  # total toll paid in each 15 minute time window
            user_info[2,user]+=vehicle_information['t_exp'][idx]  # total travel time in each 15 minute time window

        # day index, number of travelers, toll rates and total travel time for 15 min intervals
        tao_interval_information = np.vstack([day_idx, np.float32(user_info)])
        self.day+=1
        return tao_interval_information, -self.cs_list[self.day-1][0]+np.sum(self.utility['d'+str(self.day-1)]), -self.cost_list[self.day-1][0]+np.sum(self.utility['d'+str(self.day-1)]), self.cost_list[self.day-1][0], self.cs_list[self.day-1][0], self.ttcs_list[self.day-1][0]
        # social welfare, consumer surplus, generalized cost, travel cost
    

    def custgauss(self, x,mu,sigma,A):
        return A*np.exp(-(x-mu)**2/2/sigma**2)
    
    def within_day_process(self):
        # Step 1
        n=0 # Number of vehicle (accumulation)
        j=0 # index of event
        t=[] # event time
        vehicle_index=[]
        Accumulation=[]
        Speed=[]
    
        # Predicted arrival time
        Arrival_time=self.Departure_time+self.trip_len/9.78/60
    
        # Step 2
        # Define event list of departures
    
        Event_list1_array=np.zeros((self.Number_of_user,4))
        Event_list1_array[:,0]=np.arange(self.Number_of_user) # vehicle index
        Event_list1_array[:,1]=self.Departure_time # time(min)
        Event_list1_array[:,2]=np.ones(self.Number_of_user) # departure indicator: 1
        Event_list1_array[:,3]=self.trip_len # trip length
    
        # Define event list of arrivals
        Event_list2_array=np.zeros((self.Number_of_user,4))
        Event_list2_array[:,0]=np.arange(self.Number_of_user) # vehicle index
        Event_list2_array[:,1]=Arrival_time # time(min)
        Event_list2_array[:,2]=np.ones(self.Number_of_user)*2 # arrival indicator: 2
        Event_list2_array[:,3]=self.trip_len # trip length

        # S_Event_list_array: 4 columns
        # vehicle_index  time(min)  event_indicator  trip_len

        # Concatenate these two event lists
        S_Event_list_array=np.concatenate((Event_list1_array, Event_list2_array), axis=0)
    
        # Sort the list by time in ascending order
        S_Event_list_array=S_Event_list_array[S_Event_list_array[:, 1].argsort()]
    
        # get time of the first event
        t.append(S_Event_list_array[0,1]) #initial time
    
        # create a dict to store the information of each agent
        vehicle_information = {}
        vehicle_information['vehicle']=np.arange(self.Number_of_user)
        vehicle_information['trip_len(m)']=self.trip_len.astype(np.float64)
        vehicle_information['t_exp']=np.zeros(self.Number_of_user)
        vehicle_information['account']=np.zeros(self.Number_of_user)
    
        #Step 3
        # Event-based simulation
        while S_Event_list_array.shape[0]>0:
            j=j+1
            t.append(S_Event_list_array[0,1]) # record the time of the event
            if S_Event_list_array[0,2]==1:     
                vehicle_index.append(int(S_Event_list_array[0,0])) # record the agent that starts the trip

                # update the untraveled trip length
                trip_len1=vehicle_information['trip_len(m)']
                trip_len1[vehicle_index[0:-1]]=trip_len1[vehicle_index[0:-1]]-self.V(n)*60*(t[j]-t[j-1])
                vehicle_information['trip_len(m)']=trip_len1

                # update the accumulation in the network
                n=n+1
            
                # keep track of the accumulation
                Accumulation.append(n)
            
                # update the predicted arrival time
                temp=S_Event_list_array[(np.where((S_Event_list_array[:,2]==2) & (np.isin(S_Event_list_array[:,0],vehicle_index[0:-1])==True)))][:,0]
                if np.size(temp)==0:
                    temp = np.array([])
                S_Event_list_array[(np.where((S_Event_list_array[:,2]==2) & (np.isin(S_Event_list_array[:,0],vehicle_index[0:-1])==True))),1]=\
                t[j]+vehicle_information['trip_len(m)'][temp.astype(int)]/self.V(n)/60

            else: #arrival
                # update the trip length
                trip_len1=vehicle_information['trip_len(m)']
                trip_len1[vehicle_index]=trip_len1[vehicle_index]-self.V(n)*60*(t[j]-t[j-1])
                vehicle_information['trip_len(m)']=trip_len1

                # update the accumulation in the network
                n=n-1
            
                # keep track of the accumulation
                Accumulation.append(n)

                # update t_exp
                vehicle_information['t_exp'][int(S_Event_list_array[0,0])]=S_Event_list_array[0,1]-self.Departure_time[int(S_Event_list_array[0,0])]

                # remove the agent that finishes the trip
                vehicle_index.remove(int(S_Event_list_array[0,0]))
        
                # Update the predicted arrival time
                temp=S_Event_list_array[(np.where((S_Event_list_array[:,2]==2) & (np.isin(S_Event_list_array[:,0],vehicle_index)==True)))][:,0]
                if np.size(temp)==0:
                    temp = np.array([])
                S_Event_list_array[(np.where((S_Event_list_array[:,2]==2) & (np.isin(S_Event_list_array[:,0],vehicle_index)==True))),1]=\
                t[j]+vehicle_information['trip_len(m)'][temp.astype(int)]/self.V(n)/60

            # remove event from the list
            S_Event_list_array = np.delete(S_Event_list_array, (0), axis=0)    
            S_Event_list_array=S_Event_list_array[S_Event_list_array[:, 1].argsort()]
            # update speed with Speed.function
            Speed.append(self.V(n))
        vehicle_information['t_dep']=Event_list1_array[:,1]
        vehicle_information['t_arr']=vehicle_information['t_dep']+vehicle_information['t_exp']
        time_list=np.concatenate((vehicle_information['t_dep'], vehicle_information['t_arr']), axis=0)
        time_list=time_list=np.sort(time_list,axis=None)
        return vehicle_information, time_list, Accumulation, Speed

    def V(self, x):
        if isinstance(x,list):
            return [np.square(1-i/self.capacity)*9.78 for i in x]
        else:
            return np.square(1-x/self.capacity)*9.78
    
    def set_capacity(self, cap):
        self.capacity=cap

    def rearrange(self, df, cols):
    
        all_values=[]
        for idx,i in enumerate(cols):
        
            vals=df[i].T[idx].values
            all_values.append(vals)
    
        if len(cols[0])>1:
            return pd.DataFrame(data=np.vstack(all_values),columns=('t'+str(i) for i in range(2*self.tao+1)))
        else:
            return pd.DataFrame(data=np.vstack(all_values))
    
    def schedule_delay(self, T_diff):
        SD=np.empty((self.Number_of_user,2*self.tao+1))
        for i in range(2*self.tao+1):
            SD[:,i]=self.L*T_diff[:,i]*(1-np.array(T_diff[:,i]<0).astype(int))-\
            self.E*T_diff[:,i]*np.array(T_diff[:,i]<0).astype(int)
        return SD
    
    def get_day(self):
        return self.day

class CommuteEnv(Env):
    def __init__(self):
        super().__init__()
        self.params = {'alpha':1.1, 'omega':0.9, 'theta':5*10**(-1), 'tao':90, 'Number_of_user':3700} # alpha is not used and is calculated in simulation
        # define action space for each actionable value
        # mu, sigma, A
        self.action_space = Box(low=np.array([0.0, 0.0, 0.0]), high=np.array([1.0, 1.0, 1.0]), shape=(3,), dtype=np.float32)
      
        # define obs space
        self.observation_space = Box(low=np.vstack((np.full((1,15), -np.inf), np.full((1,15), -np.inf))),\
                                     high=np.vstack((np.full((1,15), np.inf), np.full((1,15), np.inf))),\
                                     shape=(2,15), dtype=np.float32)
        
        # initialise simulation
        self.sim = Simulation(self.params)
        
        # sample initial toll profile parameters
        self.mu = random.random()*90.0 #0.0 to 90.0
        self.sigma = random.random()*50.0 #0.0 to 50.0
        self.A = random.random()*20.0 #0.0 to 20.0

        self.tt_eps = []
        # social welfare, consumer surplus, generalized cost, travel cost    
        self.sw_eps = []
        self.cs_eps = []
        self.gc_eps = [] 
        self.tc_eps = []
        self.ttcs_eps = []

        self.tt_all_eps = []
        self.sw_all_eps = []
        self.cs_all_eps = []
        self.gc_all_eps = [] 
        self.tc_all_eps = []
        self.ttcs_all_eps = []

        self.first_ep = True
        self.day = 0
        
    def step(self, action):
        self.mu = action[0]*90.0
        self.sigma = action[1]*50.0
        self.A = action[2]*20.0
        
        self.day+=1
        
        action = {'mu': self.mu, 'sigma': self.sigma, 'A': self.A}

        tao_interval_information, sw, cs, gc, tc, ttcs = self.sim.step(action) # day idx, number of users, toll paid, travel time
        self.sw_eps.append(sw) # social welfare
        self.cs_eps.append(cs) # consumer surplus
        self.gc_eps.append(gc) # generalized cost
        self.tc_eps.append(tc) # travel cost
        self.ttcs_eps.append(ttcs) # travel time cost
        
        tt_mu = np.sum(tao_interval_information[3])/np.sum(tao_interval_information[1])
        reward = 100000-np.sum(tao_interval_information[3])

        self.tt_eps.append(tt_mu)
        observation = np.vstack([tao_interval_information[0],tao_interval_information[1]]) # day idx, number of users
        
        
        # needed for rl function
        info = {} # has to be dict
        
        if self.sim.get_day()==30:
            done = True
        else:
            done = False

        return observation, reward, done, info
        
        
    def reset(self):
        # reset toll
        self.mu = random.random()*90.0 #0.0 to 90.0
        self.sigma = random.random()*50.0 #0.0 to 50.0
        self.A = random.random()*20.0 #0.0 to 20.0

        if self.first_ep==False:
            self.tt_all_eps.append(np.array(self.tt_eps))
            self.sw_all_eps.append(np.array(self.sw_eps))
            self.cs_all_eps.append(np.array(self.cs_eps))
            self.gc_all_eps.append(np.array(self.gc_eps))
            self.tc_all_eps.append(np.array(self.tc_eps))
            self.ttcs_all_eps.append(np.array(self.ttcs_eps))
        
            self.tt_eps = []
            self.sw_eps = []
            self.cs_eps = []
            self.gc_eps = [] 
            self.tc_eps = []
            self.ttcs_eps = []
        
        self.first_ep = False
        self.sim = Simulation(self.params)
        observation = np.zeros((2,15))
        self.day = 0

        return observation

    def get_day(self):
        return self.day

    def get_tt(self):
        return np.array(self.tt_all_eps)

    def get_sw(self):
        return np.array(self.sw_all_eps)

    def get_cs(self):
        return np.array(self.cs_all_eps)

    def get_gc(self):
        return np.array(self.gc_all_eps)

    def get_tc(self):
        return np.array(self.tc_all_eps)

    def get_ttcs(self):
        return np.array(self.ttcs_all_eps)

    def get_toll_profile(self):
        x = np.linspace(0,15,15)
        params = {'A': self.A, 'mu': self.mu, 'sigma': self.sigma}
        toll_profile = self.sim.custgauss(x,**params)
        return toll_profile, x

    def get_params(self):
        params = {'A': self.A, 'mu': self.mu, 'sigma': self.sigma}
        return params

    def set_params(self, params):
        self.A = params['A']
        self.mu = params['mu']
        self.sigma = params['sigma']

    def set_capacity(self, cap):
        self.sim.set_capacity(cap)

    def render(self, mode):
        pass