#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Name:            MIS40550_Assgnmnt_02_V1.0
Authors:         Shane McCarthy (14200512), Richard McAleavey (15204643)
Created:         01/4/2017


Purpose:        This script forms our MIS40550 Assignment 02 submission titled:
                ‘Fake News’, Rumour Propagation and Source Inference on Social 
                Networks
                
Instructions:   Python version 3.5x or later is required to run this script, 
                there is known issues when running this script on 3.4x
                
Sections:      (1) Define classes used in simulations and experiments 
               (2) Test cases/demos 
               (3) Simulations and experiments   
               (4) Plots and visualisation  
"""
import matplotlib.pyplot as pl
import networkx as nx
import numpy as np
import random as r
import textwrap as w
from math import factorial
import pandas as pd 
import time as tm 
import os
import sys

"""
Section 1: define classes used in simulations and experiments later in the script 

"""

class Rumour_infection:
    
    """
    This is a variation of a standard SI epidemic model adapted to model the 
    propagation of a rumour through a network. We track who has or hasn’t 
    observed the rumour, however unlike a traditional SI model we’re not too 
    interested in binary state of observed/non-observed (or susceptible/ infected). 
    Instead we propose a concept where each node has an initial attitude 
    between -1 and 1 and a gullibility factor between 0 and 1, if a node 
    observes the rumour in a time period t its initial attitude is updated to 
    form an acquired attitude, a nodes acquired attitude is a function of its 
    initial attitude, gullibility/suggestibility factor and whether it observes the rumour. 
    Not all nodes that observe the rumour transmit it with its neighbours, only 
    nodes that meet a certain acquired attitude threshold share the rumour. 
    In the twitter context this could be thought of retweeting a rumour that 
    you believe to be true.
    """
    
    def __init__(self):
        pass

    
    def set_attitude(g, dstrb_ia='uniform',dstrb_gull='uniform',special_infu=False,seed=None):
        """
        This function is used to set the initial attitude (value between -1 and 1)
        and a gullibility/suggestibility factor (value between 0 and 1) of each node using 
        a random variable from a specific distribution. We’re keen to explore 
        how the distribution of initial attitude for a graph impacts how a 
        rumour is propagated through the network. This function provides the 
        option of generating normal, beta and exponential distributions for 
        the initial attitude and gullibility/suggestibility factor – used for later experiments. 

        Parameters: 
            g: a networkx graph object 
            dstrb_ia: initial attitude distribution
                - 'uniform': Uniform distribution between -1 and 0
                - 'normal':  Normal distribution with mean 0 and standard dev. .3
                - 'beta':    Beta distribution between -1 and 0
                - 'expn1':   Exponential distribution skewed to -1
                - 'exp1':    Exponential distribution skewed to 1
            dstrb_gull: gullibility/suggestibility factor distribution
                - 'uniform': Uniform distribution between 0 and 1
                - 'normal':  Normal distribution with mean .5 and standard dev. .15
            special_infu: influencer as a special case of node 
                - True|False 
        """    
        # We’re keen to explore how the distribution of initial attitude for a graph 
        # impacts how a rumour is propagated through the network. Here we generate 
        # normal, beta and exponential distribution series of experiments.  
        
        #set seed for reproducibility 
        np.random.seed(seed)
        r.seed(seed) #both numpy and random libs are being used 
        
        # distribution options for initial attitude,specified using function parameters 
        if dstrb_ia =='uniform':
            x=np.random.uniform(low=-1.0, high=1.0, size=10000)
        elif dstrb_ia =='normal':
            x= np.random.normal(0, .3, 1000)
        elif dstrb_ia =='beta':
            # The Beta distribution is a special case of the Dirichlet distribution, 
            # and is related to the Gamma distribution
            x= -1 + (np.random.beta(.4, .4,size=10000) * (1 - -1))
        elif dstrb_ia =='expn1':
            x=(-1 + (np.random.exponential(.1,10000)* (1 - -1)))
        elif dstrb_ia =='exp1':
            x=(-1 +((np.random.exponential(.1,10000)*-1)+1)* (1 - -1))
            
        # distribution options for gullibility/suggestibility,,specified using function parameters 
        if dstrb_gull =='uniform':
            y=np.random.uniform(low=0.0, high=1.0, size=10000)
        elif dstrb_gull =='normal':
            y= np.random.normal(.5, .15, 1000)
    
        # apply to graph nodes
        for i in g.nodes():
            #Give each node and id
            g.node[i]['id']=i
            g.node[i]['initial_attitude']=r.choice(x)
            # We initialise acquired attitude to 0 at t=0
            g.node[i]['acquired_attitude']=0
            g.node[i]['gullibility']=r.choice(y)
    
        # Next we define an influencer as a special case of node, these nodes
        # have a fixed attitude and are not open minded so gullibility/suggestibility=0
        # by definition they have high influence in the network, here we use degree
        # centrality as the measure of influence (alternatives could be argued)
        # To be classed as a influence a node has to be in the top 10% of degree
        # centrality for the network 
        
        if special_infu == True: 
            inf_nodes=int(round(len(g.nodes())*.1))
            inf_measure=nx.degree_centrality(g)
            top_infs=sorted(inf_measure, key=inf_measure.get, reverse=True)[:inf_nodes]
            for n in top_infs:
                g.node[n]['initial_attitude']=1
                g.node[n]['gullibility']=0

        return g

    def initialise_rumour_source(G,n_nodes,sel_source=None,n_sources=1,rumour=1,seed=None):
        """
        Function to initialise rumour source(s)
        For the nodes in the graph the function selects a specified number of 
        random source node(s). The function then assigns the source node(s) 
        with the rumour - a value between -1 and 1, typically for experimental 
        purposes we assign -1 or 1. The inputs for the rumour_step function 
        are then prepared.For all experiments in this assignment we’ll use one 
        random source as multiple source problems are outside scope. 
        
         Parameters: 
             G:             Input graph
             n_nodes:       Number of nodes in graph 
             n_sources:     Number of sources typically 1
             sel_source:    Used to manually select the source, if = None a 
                            source is selected at random
             rumour:        Rumour value typically 1 or -1 
             seed:          For repeatable random selection 
        
        """
        
        #set seed for reproducibility 
        np.random.seed(seed)
        
        #we can select a source manually with the sel_source parameter, if set 
        #to None a random source is selected.
        if sel_source==None: 
        #select a random node(s) to act as the source of the rumour
            source =(np.random.choice(range(0,n_nodes), n_sources,replace=False)).tolist()
        else:
            source=[sel_source]

        #graph object to keep neighbours of nodes infected with rumour 
        frontier = nx.Graph()
        
        #We use “infected” over “observed” to remove ambiguity  
        #graph object to keep nodes infected with rumour 
        
        infected = nx.Graph()
        #read source node into infected graph object
        for source in source:
            infected.add_node(source) 
            infected.node[source]['initial_attitude']=0
            infected.node[source]['acquired_attitude']=rumour
            infected.node[source]['gullibility']=0

        for node in infected:
            for neighbor in G.neighbors(node):
                if not infected.has_node(neighbor):
                    frontier.add_node(neighbor)
        
        return source, infected, frontier
    
    def rumour_step(G,frontier, infected, prob,rumour,trnsmt_tshd):
            """ 
            This is the step function for the rumour, it generates a step in 
            the rumour simulation. For simplicity (and the requirements of 
            experiments) two primary graph objects are maintained, 
            G is the original graph and "infected is subgraph who have 
            observed the rumour.  

            Parameters: 
                G: a networkx graph object
                frontier: graph object to store neighbours of nodes that observed rumour
                infected: graph object to store nodes that have observed rumour
                prob:     probability of observing rumour at time t 
                rumour:   We assume the rumour can be represented as a 
                          single real number between -1 and 1
                Trnsmt_tshd: The minimal acquired attitude required to transmit the rumour  
            """
            # List containing the nodes from the frontier at time t with k
            # neighbors who have obserevd the rumour
            frontier_degree_t = []
            # List containing the newly obserevd nodes
            infected_t = []
            #List of nodes that have obserevd and transmitted the rumour 
            transmitted = []

            # Generating the lists containing the nodes with i obserevd neighbors
            for node in frontier:
                i = 0
                for neighbor in G.neighbors(node):
                    if infected.has_node(neighbor):
                        i += 1
                try:
                    frontier_degree_t[i-1].append(node)
                except IndexError:
                    for j in range(0, i+1):
                        t = []
                        frontier_degree_t.append(t)
                    frontier_degree_t[i-1].append(node)
    
            # Generating the optimal step for the rumour
            for j in range(0, len(frontier_degree_t)):
                if len(frontier_degree_t[j]) > 0:
                    f_j = len(frontier_degree_t[j])
                    print('Number of neighbors that have not heard rumour:',f_j)
                    p_j = 1-(1-prob)**(j+1)
                    print('Probability of rumour being transmitted:',p_j)
                    n_j = int(min(np.floor(p_j*(f_j+1)), f_j))
                    print('Number of neighbors that will hear the rumour:',n_j)
                    infected_t.append(np.random.choice(frontier_degree_t[j], n_j,replace=False))
                        
            # Updating the frontier
            for j in infected_t:
                # Updating each node that has observed the rumour
                for node in j:
                    infected.add_node(node)
                    infected.node[node]['initial_attitude']=G.node[node]['initial_attitude']
                    infected.node[node]['gullibility']=  G.node[node]['gullibility']
                    #This is the acquired_attitude calculation, it is capped so the value can't exceed 1 
                    infected.node[node]['acquired_attitude']=  min(((G.node[node]['initial_attitude'])
                                                        +((G.node[node]['gullibility'])*(rumour))),1)
                    
                    #Once added to the infected graph we remove from the frontier
                    frontier.remove_node(node)
                    #Only if the newly infected node exceedes acquired attitude treshold can it transmitt the rumour
                    if infected.node[node]['acquired_attitude'] > trnsmt_tshd:
                        transmitted.append(node)
                        for neighbor in G.neighbors(node):
                                if not infected.has_node(neighbor):
                                    frontier.add_node(neighbor) 
                                    
            return infected_t,transmitted               

    def simulation(G,frontier, infected,rumour,time_periods,prob,trnsmt_tshd,graph_plot=False,diag_plot=False,save_plot=False,labels=True):
        """ 
        Simulation function, iterates the step function until the graph is 
        fully converged (all nodes have heard rumour) OR no node that has 
        observed the rumour has an “acquired attitude” over the threshold 
        permitted to transmit the rumour to its neighbours. The threshold for 
        this set of simulations was set to 0.25, recalling that the spectrum of 
        attitude is between -1 and 1 this threshold is relatively low. A step 
        limit is also applied to avoid endless loops

        This function provides the option of visual output (network plots and 
        simulation diagnostic plots). This is useful for validating the 
        simulation and understanding the model however it is not practical for 
        very large graphs or when running multiple simulations. 

        
        Parameters: 
                G:              a networkx graph object
                frontier:       graph object to store neighbours of nodes that observed rumour
                infected:       graph object to store nodes that have observed rumour
                rumour:         We assume the rumour can be represented as a 
                                single real number between -1 and 1
                time_periods:   t periods to run the simulation for
                prob:           probability of observing the rumour at time t 
                trnsmt_tshd:    the minimal acquired attitude required to 
                                transmit the rumour 
                graph_plot:     True|False - for each t a graph differentiating 
                                  nodes that have/haven’t observed the rumour is plotted 
                diag_plot:      True|False - display simulation diagnostic plots
                save_plots:     True|False - save plots to file.
        """
 
        t=0                 # initialize time step to 0, increment by 1 for each step
        aa_delta_t=[]       # list to store avarge acquired attitude per t
        rumour_prop=[]      # Proportion of nodes who have heard rumour
        pass_prop=[]        # Proportion of nodes that transmitted rumour
        not_infected=[1]    
        # We want to run the simulation until all nodes are infected or simulation
        # times out initialize not_infected with 1 as while needs 
        #len(not_infected) > 0 to start
        pos = nx.fruchterman_reingold_layout(G) # keep node position fixed
    
        while len(not_infected) > 0 and t < time_periods:
            # run simulation until all nodes have been infected or it times out 
            
            #Transposes the edges from graph to graph_i
            Rumour_infection.edging(G, infected)
            
            if graph_plot==True: # Only plot graphs if True
                #set up plot
                col=list((nx.get_node_attributes(infected,'acquired_attitude')).values())
                fig=pl.figure(num=None, figsize=(8, 8), dpi=80)
                ax = fig.add_subplot(111)
                
                #draw orginal graph G + infected graph
                nx.draw(G, pos=pos,with_labels= labels,font_size=8, font_color='black'
                        ,node_color = 'seagreen')
                
                nx.draw(infected, pos=pos,edge_color='lightcoral',font_size=8, 
                        font_color='black',font_weight='bold',
                        with_labels= labels,node_color=col,cmap=pl.cm.Blues)
                
                #Add timestamp t to plot
                text1 ='Time step:'+str(t)#+'- the infected nodes are:'+str(infected.nodes())
                ax.text(0.01, 0.000002, (w.fill(text1, 100)), verticalalignment='bottom', 
                        horizontalalignment='left', transform=ax.transAxes,color='black', 
                        fontsize=10,wrap=True)
                
                if save_plot==True: 
                    pl.savefig("rumour_%s.png" % str(t).zfill(3),dpi=1000) # pad filename with zeros
                pl.show()
                print('Time step:',t)
                print('The infected nodes are:',infected.nodes())
            
            
            # call rumour_step 
            not_infected,transmitted =Rumour_infection.rumour_step(G, frontier,
                                                   infected, prob,1,trnsmt_tshd)
            t=t+1 #Increment t
            
            #Logic to measure the mean acquired attitude as t increase 
            #First get the acquired attitude of nodes that have obeserved the 
            #rumour at each t 
            aa_infected=nx.get_node_attributes(infected,'acquired_attitude')
            #we alos have to get acquired attitude of nodes that have NOT 
            #obeserved the rumoußr at each t 
            aa_all=nx.get_node_attributes(G,'acquired_attitude')
            not_infected_at_t = [val for sublist in not_infected for val in sublist]
            aa_not_infected = { not_infected_at_t: aa_all[not_infected_at_t] 
                               for not_infected_at_t in not_infected_at_t }
            aa_both={**aa_infected,**aa_not_infected}
            avg_aa=sum(list(aa_both.values()))/float(len(list(aa_both.values())))
            aa_delta_t.append(avg_aa)
            rumour_prop.append(len(infected.node)/len(G.node))
            pass_prop.append(len(transmitted)/len(infected.node))
            #histogram of intial initial attitude     
            
        if diag_plot==True: # Only plot simulation diagnostics if True
            Rumour_infection.diagnostic_plots(G,infected,aa_delta_t,rumour_prop,pass_prop)

        return G, infected,t,aa_delta_t,rumour_prop,pass_prop, pos
        
  
        
    def edging(graph, graph_i):
        """ 
        Simple function to transposes the edges from the orginal graph G to
        the infceted graph
        """
        for node in graph_i:
            for neighbor in graph.neighbors(node):
                if graph_i.has_node(neighbor):
                    graph_i.add_edge(node, neighbor)
        
    def diagnostic_plots(G,infected,aa_delta_t,rumour_prop,pass_prop): 
        """ 
        This is a simple function to call a number of diagnostic plots following 
        the run of a simulation. 

        Parameters: 
                G:              orginal graph object 
                infected:       graph object to store nodes that have observed rumour
                aa_delta_t:     mean acquired attitude at each t of simulation
                rumour_prop:    % of nodes who have heard rumour
                pass_prop:      % of nodes who heard and transmitted  the rumour            
        """
        
        pl.figure(figsize=(8,5))
        pl.title('Distribution of initial attitude') 
        pl.hist(list((nx.get_node_attributes(G,'initial_attitude')).values()),
                50,normed=True) 
        pl.ylabel('Proportion %')
        pl.xlabel('initial attitude (0-1)')
        pl.show()
        
        pl.figure(figsize=(8,5))
        pl.title('Distribution of gullibility') 
        pl.hist(list((nx.get_node_attributes(G,'gullibility')).values()), 50, 
                 normed=True)
        pl.ylabel('Proportion %')
        pl.xlabel('gullibility (0-1)')
        pl.show()
        pl.figure(figsize=(8,5))
        pl.title('Distribution of acquired attitude') 
        pl.hist(list((nx.get_node_attributes(infected,
        'acquired_attitude')).values()), 50, normed=True)
        pl.ylabel('Proportion %')
        pl.xlabel('acquired attitude (0-1)')
        pl.show()
        
        pl.figure(figsize=(8,5))
        pl.title('Distribution of acquired attitude') 
        pl.plot(aa_delta_t)
        pl.ylabel('acquired attitude (0-1)')
        pl.xlabel('Time Step')
        pl.show()
        
        pl.figure(figsize=(8,5))
        pl.title('Rumour Propagatation: % of nodes who have heard rumour') 
        pl.plot(rumour_prop)
        pl.ylabel('% of nodes who have heard rumour')
        pl.xlabel('Time Step')
        pl.show()
        
        pl.figure(figsize=(8,5))
        pl.title('Rumour Propagatation: % of nodes who heard and transmitted the rumour') 
        pl.plot(pass_prop)
        pl.ylabel('% of nodes who heard and transmitted the rumour')
        pl.xlabel('Time Step')
        pl.show()

        
        
class Source_inference:
    """
    This source inference class is primarily based off the work of Shah and Zaman
    Their approach has been adapted to deal with larger graphs. This class has 
    been tested for random graphs up to 2000 nodes and a real facebook graph of 500 nodes. 
    
    One clear limitation is that we’re using int to store factorial values, 
    factorial values are known for getting very large however using decimal 
    slows down the algorithm. For larger graphs we’ll have to address this 
    implementation challenge. 

    References
    ----------
    Shah, D. and Zaman, T., 2011. Rumors in a network: Who's the culprit?. 
    IEEE Transactions on information theory, 57(8), pp.5163-5181.
    """

    def __init__(self):
        pass

    def run(graph, i_graph):
        """
        This function executes the source inference algorithm returning the 
        most probable source and a list of all possible sources with their 
        corresponding rumour centrality. 
         Parameters: 
                graph:              orginal graph object 
                i_graph:            infceted graph

        """

        #Initialise variables to store values
        max_rumor_centrality = 0
        rumor_source = None
        rumor_centrality_all=[]
        
        
        # BFS heuristic - the intuition is that if v was the source, then the 
        # BFS tree would correspond to the fastest (intuitively, most likely) 
        #spread of the rumor
        for v in i_graph:
            tree = nx.bfs_tree(i_graph, source=v)

            rumor_centrality = Source_inference.algorithm_tree(tree, v)
            #Append the rumour centrality to infected nodes graph
            i_graph.node[v]['rumor_centrality']=rumor_centrality
            
            #capture a list of all possible sources with their corresponding 
            #rumour centrality 
            rumor_centrality_all.append(rumor_centrality)
            
            #capture most likely source
            if rumor_centrality > max_rumor_centrality:
                max_rumor_centrality = rumor_centrality
                rumor_source = v
            
        return rumor_source,rumor_centrality_all

    def algorithm_tree(tree, v=None):
        """
        This function evaluates rumor source in a tree by computing rumor 
        centrality for each node if provides an exact estimation for regular 
        trees, heuristic approach for general trees
        
            Parameters: 
                tree:              bfs_tree
                v:                 node in infected graph
        
        """
        if v is None:
            v = tree.nodes()[0]  

        # Call the permitted_permutations function 
        r = Source_inference.compute_permitted_permutations(tree, v)

        # Find the node with maximum rumor centrality
        max_rumor_centrality = 0
        #source_estimation = None
        for node in tree:
            if r[node] > max_rumor_centrality:
                #source_estimation = node
                max_rumor_centrality = r[node]
        #return source_estimation
        return r[v]

    def compute_permitted_permutations(origin_tree, v):
        """
        This function calculates the permitted permutations for the rumour graph. 
        Because of the tree structure of the graph there is a unique sequence of 
        nodes for the rumour to spread to each node in the original graph G.

            Parameters: 
                origin_tree:              bfs_tree
                v:                        node in infected graph

        """
        # We need to traverse the tree in depth order : down-up and up-down
        tree = list(nx.dfs_postorder_nodes(origin_tree))
        t = {}
        p = {}
        r = {}

        # Down-up pass
        for u in tree:
            if origin_tree.neighbors(u) == []:  # u is a leaf
                t[u] = 1
                p[u] = 1
            else:
                t[u] = sum(t[child] for child in origin_tree.successors(u)) + 1
                p[u] = t[u]
                for child in origin_tree.successors(u):
                    p[u] *= p[child]

        # Up-down pass
        tree.reverse()
        N = origin_tree.number_of_nodes()  # or len(origin_tree)
        for u in tree:
            if u == v:
                r[v] = factorial(N)
                for child in origin_tree.successors(v):
                    r[v] //= p[child]
            else:
                r[u] = r[origin_tree.predecessors(u)[0]] * t[u] // (N - t[u])

        return r
        
        
    def source_viz(G,infected,seed,source_estimation,delta,pos):
        """
        This function plots a heat map of rumour centrality 
        
        Parameters: 
        
            G:                  a networkx graph object
            infected:           graph object to store nodes that have observed rumour
            seed:               the actual source 
            source_estimation:  the estimated source
            delta:              error between actual and estimate 
            pos:                node positions from simulation plots 

        """
            
            # make the plot look pretty 
            #pos = nx.spring_layout(G)
            
            # create a heat map style plot for rumour centrality, strongest at 
            #the estimated source (highest rumour centrality) fading away from 
            #this as rumour centrality decreases 
            col=list((nx.get_node_attributes(infected,'rumor_centrality')).values())

            #normalise rumour centrality value for plotting  
            norm_col = []
            for x in col:
                norm_col.append(x//max(col))
                
                
            fig=pl.figure(num=None, figsize=(8, 8), dpi=100)
            ax = fig.add_subplot(111)
                
            nx.draw(G, pos=pos,with_labels= True,font_size=8, font_color='black'
                        ,node_color = 'seagreen') 
            
            nx.draw(infected, pos=pos,edge_color='lightcoral',font_size=8, 
                    font_color='black',with_labels= True,node_color=col,cmap=pl.cm.YlOrRd)
            
            nx.draw_networkx_nodes(infected,pos=pos, nodelist=[seed], node_color="lime",with_labels = True)
            
            #Add source, estimate, error to plot
            text1 ='Actual source:'+str(seed)+', Predicted source:'+str(source_estimation)+', Error (number of hops):'+str(delta)
            ax.text(0.01, 0.02, (w.fill(text1, 100)), verticalalignment='bottom', 
                    horizontalalignment='left', transform=ax.transAxes,color='black', 
                    fontsize=10,wrap=True)
            pl.savefig("source_inf.png",dpi=1000) # pad filename with zeros
            pl.show()
            

            
class Graph_generators:
    """
    This class consists of the following random graph generators:

    Gragh generators 
        - Erdős-Rényi graph
        - Barabási-Albert preferential attachment model
        - Watts-Strogatz small-world graph
        
    """
    
    def __init__(self):
        pass     
        
    def ER_random_graph(n,p,seed=None):    
        """
        Erdős-Rényi graph generator
        
        Parameters:
            n: Number of nodes
            p: Probability for edge creation
           seed: Seed for random number generator (default=None).
        
        
        """
        r.seed(seed)
        
        def rand(): 
        	random_int = r.randint(0,1000)
        	f_rand = float(random_int)
        	d_rand = f_rand * 0.001
        	return d_rand
    
    # build graph
        g = nx.Graph()
        for x in range (0, n):
         g.add_node(x)
        
        for x in range (0, n):
        	for y in range (x, n):
        		if (rand() < p or p == 1.0) & (x != y):
                         g.add_edge(x, y)# weight=rand())
        return g
    
        
    def barabasi_albert_graph(n, m, seed=None):
        
        """
        Return random graph using Barabási-Albert preferential attachment model.
        Modified from Networkx implementation
        
        Parameters:
            n: Number of nodes
            m: Number of edges to attach from a new node to existing nodes
           seed: Seed for random number generator (default=None).
    
        References
        ----------
        https://networkx.github.io/documentation/development/_modules/networkx/
        generators/random_graphs.html#barabasi_albert_graph
        """
        def _random_subset(seq,m):
            """ Return m unique elements from seq.
            This differs from random.sample which can return repeated
            elements if seq holds repeated elements.
            """
            targets=set()
            while len(targets)<m:
                x=r.choice(seq)
                targets.add(x)
            return targets
        
        if m < 1 or  m >=n:
            raise nx.NetworkXError(\
                  "Barabási-Albert network must have m>=1 and m<n, m=%d,n=%d"%(m,n))
        if seed is not None:
            r.seed(seed)
    
        # Add m initial nodes (m0 in barabasi-speak)
        G=nx.empty_graph(m)
        G.name="barabasi_albert_graph(%s,%s)"%(n,m)
        # Target nodes for new edges
        targets=list(range(m))
        # List of existing nodes, with nodes repeated once for each adjacent edge
        repeated_nodes=[]
        # Start adding the other n-m nodes. The first node is m.
        source=m
        while source<n:
            # Add edges to m nodes from the source.
            G.add_edges_from(zip([source]*m,targets))
            # Add one node to the list for each new edge just created.
            repeated_nodes.extend(targets)
            # And the new node "source" has m edges to add to the list.
            repeated_nodes.extend([source]*m)
            # Now choose m unique nodes from the existing nodes
            # Pick uniformly from repeated_nodes (preferential attachement)
            targets = _random_subset(repeated_nodes,m)
            source += 1
        return G

    
    def watts_strogatz_graph(n, k, p, seed=None):
        """
        Return a Watts-Strogatz small-world graph.
        Modified from Networkx implementation
        
        Parameters

            n : The number of nodes
            k : Each node is connected to k nearest neighbors in ring topology
            p : The probability of adding a new edge for each edge
            seed : seed for random number generator (default=None)
        
    
        References
        ----------
        https://networkx.github.io/documentation/development/_modules/networkx/
        generators/random_graphs.html#barabasi_albert_graph
        """
        if k>=n:
            raise nx.NetworkXError("k>=n, choose smaller k or larger n")
        if seed is not None:
            r.seed(seed)
    
        G = nx.Graph()
        G.name="watts_strogatz_graph(%s,%s,%s)"%(n,k,p)
        nodes = list(range(n)) # nodes are labeled 0 to n-1
        # connect each node to k/2 neighbors
        for j in range(1, k // 2+1):
            targets = nodes[j:] + nodes[0:j] # first j nodes are now last in list
            G.add_edges_from(zip(nodes,targets))
        # rewire edges from each node
        # loop over all nodes in order (label) and neighbors in order (distance)
        # no self loops or multiple edges allowed
        for j in range(1, k // 2+1): # outer loop is neighbors
            targets = nodes[j:] + nodes[0:j] # first j nodes are now last in list
            # inner loop in node order
            for u,v in zip(nodes,targets):
                if r.random() < p:
                    w = r.choice(nodes)
                    # Enforce no self-loops or multiple edges
                    while w == u or G.has_edge(u, w):
                        w = r.choice(nodes)
                        if G.degree(u) >= n-1:
                            break # skip this rewiring
                    else:
                        G.remove_edge(u,v)
                        G.add_edge(u,w)
        return G
        
    
        
""""
 Section two: Test cases/demos 
 
""""


# Test case 1
# barabasi_albert_graph n=200
# randomly selected source 

G = Graph_generators.barabasi_albert_graph(n=200,m=1,seed=4352)
G_a=Rumour_infection.set_attitude(g=G,dstrb_ia='beta',dstrb_gull='uniform',special_infu=False,seed=999)
seed, infected, frontier=Rumour_infection.initialise_rumour_source(G=G_a,n_nodes=200,n_sources=1,rumour=1,seed=9876)
G_b, G_i,t,aa_delta_t,rumour_prop,pass_prop=Rumour_infection.simulation(G_a,frontier,infected,rumour=1, time_periods=50,prob=.7,trnsmt_tshd=.25,graph_plot=True,diag_plot=True,save_plot=False)
source_estimation,rumor_centrality_all = Source_inference.run(G, G_i)
delta=nx.dijkstra_path_length(G, seed, source_estimation)
Source_inference.source_viz(G,G_i,seed,source_estimation,delta)


# Test case 2
# watts_strogatz_graph n=200
# manually selected source 

G = Graph_generators.watts_strogatz_graph(n=150,k=2,p=.7,seed=4352)
G_a=Rumour_infection.set_attitude(g=G,dstrb_ia='exp1',dstrb_gull='uniform',special_infu=False,seed=1234)
seed, infected, frontier=Rumour_infection.initialise_rumour_source(G=G_a,n_nodes=150,sel_source=0,n_sources=1,rumour=1,seed=9876)
G_b, G_i,t,aa_delta_t,rumour_prop,pass_prop=Rumour_infection.simulation(G_a,frontier,infected,rumour=1, time_periods=50,prob=.7,trnsmt_tshd=.25,graph_plot=True,diag_plot=True,save_plot=False)
source_estimation,rumor_centrality_all = Source_inference.run(G, G_i)
delta=nx.dijkstra_path_length(G, seed, source_estimation)
Source_inference.source_viz(G,G_i,seed,source_estimation,delta)


# Test case 3
# Erdős-Rényi graph n=200
# randomly selected source 

G = Graph_generators.ER_random_graph(n=50,p=.1,seed=4352)
G_a=Rumour_infection.set_attitude(g=G,dstrb_ia='beta',dstrb_gull='uniform',special_infu=False,seed=1234)
seed, infected, frontier=Rumour_infection.initialise_rumour_source(G=G_a,n_nodes=50,sel_source=None,n_sources=1,rumour=1,seed=9876)
G_b, G_i,t,aa_delta_t,rumour_prop,pass_prop=Rumour_infection.simulation(G_a,frontier,infected,rumour=1, time_periods=50,prob=.7,trnsmt_tshd=.25,graph_plot=True,diag_plot=True,save_plot=False)
source_estimation,rumor_centrality_all = Source_inference.run(G, G_i)
delta=nx.dijkstra_path_length(G, seed, source_estimation)
Source_inference.source_viz(G,G_i,seed,source_estimation,delta)


# Test case 4 - Real facebook graph from http://snap.stanford.edu/data/index.html
# randomly selected source 
G_fb = nx.read_edgelist("facebook_combined.txt", create_using = nx.Graph(), nodetype = int)
G_a=Rumour_infection.set_attitude(g=G_fb,dstrb_ia='beta',dstrb_gull='uniform',special_infu=False,seed=483)
seed, infected, frontier=Rumour_infection.initialise_rumour_source(G=G_a,n_nodes=nx.number_of_nodes(G_fb),sel_source=0,n_sources=1,rumour=1,seed=None)
G_b, G_i,t,aa_delta_t,rumour_prop,pass_prop=Rumour_infection.simulation(G_a,frontier,infected,rumour=1, time_periods=50,prob=.7,trnsmt_tshd=.25,graph_plot=True,diag_plot=True,save_plot=True,labels=False)
source_estimation,rumor_centrality_all = Source_inference.run(G_fb, G_i)

#write to gephi file for better Visualisation 
nx.write_gexf(G_fb, "fb_graph.gexf")


""""
 Section Three: Simulations and experiments   

""""           
#Experiment 1: Rumour propagation and source inference
# In this simulation we want to develop a deeper understanding of how our rumour 
# propagation model performs on Erdős-Rényi, Barabasi & Albert  Watts & Strogatz  
# graphs of different size and with different distributions of initial attitude 



    def find_source_exp(G_choice,n_start,n_end,increment,repeat):
        """
        This simulation constructs specific random graphs, with an increasing 
        number of nodes and different initial attitude distributions. A random 
        node is then selected as the source/seed of a rumour and our model is 
        used to propagate the rumour through the network. The source inference 
        algorithm then attempts to find the source (obviously without being 
        told the true source. This process is repeated multiple times for each 
        random graph, of specific size and initial attitude distribution to 
        explore how both models behave in general. 
        
        Parameters

            G_choice :  Erdős-Rényi 'er', Barabasi & Albert 'ba' and  Watts & Strogatz 'sw'
            n_start :   Graph size start from  
            n_end :     Graph size end before
            increment : increment step
            repeat :    Number of times to repeat rumour propagation and source 
                        inference for each random graph of size n 
        """

        #Lists to pick up results
        d_n=[]
        t_n=[]
        n_n=[]
        aa_n=[]
        rumour_prop_n=[]
        pass_prop_n=[]  
        delta_src_est_n =[] 
        si_n=[]

        # Initial attitude distributions
        dists = ['normal', 'beta', 'exp1', 'uniform']
    
        # Loop across all 4 distributions
        for dist in dists:  
            # Loop for each increment in graph size
            for n_size in range(n_start, n_end,increment):

                    d_s=[]
                    n_s=[]
                    t_t=[]
                    aa_t=[]
                    si_t=[]
                    rumour_prop_t=[]
                    pass_prop_t=[]
                    delta_src_est =[]
      
            
                    
                    # We want to repeat multiple times for each graph of size n
                    # to to explore how both models behave in general.
                    # Note the random seeds are set to “None” 
                    
                    for i in range(repeat):
                        
                        if G_choice =='er':
                             G=Graph_generators.ER_random_graph(n=n_size,p=.1,seed=None)
                        elif G_choice =='ba':
                             G= Graph_generators.barabasi_albert_graph(n=n_size,m=1,seed=None)
                        elif G_choice =='sw':
                             G=Graph_generators.watts_strogatz_graph(n=n_size,k=2,p=.7,seed=None)
                            
                        G_a=Rumour_infection.set_attitude(g=G,dstrb_ia=dist,dstrb_gull='uniform',special_infu=False,seed=None)
                        source_seed, infected, frontier=Rumour_infection.initialise_rumour_source(G=G_a,n_nodes=n_size,sel_source=None,n_sources=1,rumour=1,seed=None)
                        G_b, G_i,t,aa_delta_t,rumour_prop,pass_prop=Rumour_infection.simulation(G_a,frontier,infected,rumour=1, time_periods=50,prob=.7,trnsmt_tshd=.25,graph_plot=False,diag_plot=False,save_plot=False)
                        start=tm.time()
                        source_estimation,rumor_centrality_all = Source_inference.run(G_a, G_i)
                        end=tm.time()
                        delta=nx.dijkstra_path_length(G_a, source_seed, source_estimation)
                        
                        

                        # Vector of t(s), i.e. how long did it take each simulation to
                        # stop propagating due to convergence (all nodes have heard it) 
                        # or the rumour died 
                        t_t.append(t-1)
                        # Acquired attitudes, what was the mean acquired attitude for 
                        # each simulation when simulation ended/rumour stopped propagating
                        aa_t.append(float(aa_delta_t[t-1]))
                        # Heard proportion, for each simulation what proportion of nodes heard 
                        #the rumour by the time the simulation ended/rumour stopped propagating
                        rumour_prop_t.append(float(rumour_prop[t-1]))
                        # Transmitted proportion, for each simulation what proportion of 
                        #nodes heard and transmitted the rumour. Here we take the mean across e
                        #ach time period 
                        pass_prop_t.append((sum(pass_prop)/float(len(pass_prop))))
                        # What proportion of the observed transmitted the rumour for each simulation 
                        delta_src_est.append(delta)
                        d_s.append(dist)
                        n_s.append(n_size)
                        si_t.append((end-start))
                        
                    si_n.append(si_t)    
                    d_n.append(d_s)
                    n_n.append(n_s)  
                    t_n.append(t_t)
                    aa_n.append(aa_t)
                    rumour_prop_n.append(rumour_prop_t)
                    pass_prop_n.append(pass_prop_t)
                    delta_src_est_n.append(delta_src_est)
                    print(n_size)
                             
                    
        results = pd.DataFrame({'Initial_attitude_dist': sum(d_n, []),'Nodes': 
                 sum(n_n, []), 'Time': sum(t_n, []),'Attitude': sum(aa_n, []),
                 'Rumour_Prop': sum(rumour_prop_n, []),'Transmitt_Prop': sum(pass_prop_n, []), 
                 'Delta': sum(delta_src_est_n, []),'SI_time': sum(si_n, [])})
        
        return results
        
#Experiment 2: Rumour propagation and source inference
# In this simulation we want to develop a deeper understanding of how our rumour 
# propagation model performs and source inference models work on a real graph
        
def find_source_real(G,repeat):
        """
       This simulation uses a real Facebook graph initialised with different 
       initial attitude distributions. A random node is then selected as the 
       source/seed of a rumour and our model is used to propagate the rumour 
       through the network. The source inference algorithm then attempts to 
       find the source. This process is repeated multiple times for each random 
       source and initial attitude distribution to explore how both models 
       behave in general.

        
        Parameters

            G :         Input graph
            repeat :    Number of times to repeat rumour propagation and source 
                        inference 
        """

        #List to pick up results
        d_n=[]
        t_n=[]
        n_n=[]
        aa_n=[]
        rumour_prop_n=[]
        pass_prop_n=[]  
        delta_src_est_n =[] 
        si_n=[]

        # Initial attitude distributions
        dists = ['normal', 'beta', 'exp1', 'uniform']
    
        # Loop across all 4 distributions
        for dist in dists:  


                    d_s=[]
                    n_s=[]
                    t_t=[]
                    aa_t=[]
                    si_t=[]
                    rumour_prop_t=[]
                    pass_prop_t=[]
                    delta_src_est =[]
      
            
                    
                    # We want to repeat multiple times for each graph of size n
                    # to to explore how both models behave in general.
                    # Note the random seeds are set to “None” 
                    
                    for i in range(repeat):
                        
                        G_a=Rumour_infection.set_attitude(g=G,dstrb_ia=dist,dstrb_gull='uniform',special_infu=False,seed=None)
                        source_seed, infected, frontier=Rumour_infection.initialise_rumour_source(G_a,n_nodes=nx.number_of_nodes(G),sel_source=r.choice(G.nodes()),n_sources=1,rumour=1,seed=None)
                        G_b, G_i,t,aa_delta_t,rumour_prop,pass_prop=Rumour_infection.simulation(G_a,frontier,infected,rumour=1, time_periods=50,prob=.7,trnsmt_tshd=.25,graph_plot=False,diag_plot=False,save_plot=False)
                        start=tm.time()
                        source_estimation,rumor_centrality_all = Source_inference.run(G_a, G_i)
                        end=tm.time()
                        delta=nx.dijkstra_path_length(G_a, source_seed, source_estimation)
                        
                        

                        # Vector of t(s), i.e. how long did it take each simulation to
                        # stop propagating due to convergence (all nodes have heard it) 
                        # or the rumour died 
                        t_t.append(t-1)
                        # Acquired attitudes, what was the mean acquired attitude for 
                        # each simulation when simulation ended/rumour stopped propagating
                        aa_t.append(float(aa_delta_t[t-1]))
                        # Heard proportion, for each simulation what proportion of nodes heard 
                        #the rumour by the time the simulation ended/rumour stopped propagating
                        rumour_prop_t.append(float(rumour_prop[t-1]))
                        # Transmitted proportion, for each simulation what proportion of 
                        #nodes heard and transmitted the rumour. Here we take the mean across e
                        #ach time period 
                        pass_prop_t.append((sum(pass_prop)/float(len(pass_prop))))
                        # What proportion of the observed transmitted the rumour for each simulation 
                        delta_src_est.append(delta)
                        d_s.append(dist)
                        si_t.append((end-start))
                        
                    si_n.append(si_t)    
                    d_n.append(d_s)
                    t_n.append(t_t)
                    aa_n.append(aa_t)
                    rumour_prop_n.append(rumour_prop_t)
                    pass_prop_n.append(pass_prop_t)
                    delta_src_est_n.append(delta_src_est)
                    print(dist)
                    print(i)
                             
                    
        results = pd.DataFrame({'Initial_attitude_dist': sum(d_n, []), 'Time': sum(t_n, []),'Attitude': sum(aa_n, []),'Rumour_Prop': sum(rumour_prop_n, []),'Transmitt_Prop': sum(pass_prop_n, []), 'Delta': sum(delta_src_est_n, []),'SI_time': sum(si_n, [])})
        
        return results
        
#run simulations 
ba_sim = find_source_exp('ba',100,2100,100,20)        
sw_sim = find_source_exp('sw',100,2100,100,20)        
er_sim = find_source_exp('er',100,1000,100,20)

#save down results
#ba_sim.to_csv('ba_sim.csv', sep=',')
#sw_sim.to_csv('sw_sim.csv', sep=',')
#er_sim.to_csv('er_sim.csv', sep=',')

#read in facebook graph
G_fb = nx.read_edgelist("facebook_combined_sample500.txt", create_using = nx.Graph(), nodetype = int)

#run simulations 
fb_stats_a =find_source_real(G_fb,100)
#save down results
fb_stats_a.to_csv('fb_stats_all.csv', sep=',')

""""
 Section four: Plots and visualisations  

""""
#excuse the import statement here, just requires to get nice ggplot graphs! 
import matplotlib
matplotlib.style.use('ggplot')

#Barabasi & Albert plots 

x=ba_sim.loc[ba_sim['Initial_attitude_dist'] != 'x']['Delta']
df=x.to_frame(name='error')
axis=(max(df['error']))+2
pl.figure(figsize=(10,7))
pl.title('Barabasi & Albert graph: 1600 Simulations, 100-2000 nodes',fontsize=12)
df['error'].hist(color='seagreen',bins=np.arange(axis),normed=True,xlabelsize =10)
pl.ylabel('Proportion')
pl.xlabel('Error between actual source and inferred source (number of hops)')
pl.savefig("ba_ALL.png",dpi=500) 
pl.show()


#Barabasi & Albert plots 
# Initial attitude distributions
dists = ['normal', 'beta', 'exp1', 'uniform']
# Loop across all 4 distributions
for dist in dists:  
    x=ba_sim.loc[ba_sim['Initial_attitude_dist'] == dist]['Delta']
    df=x.to_frame(name='error')
    pl.figure(figsize=(10,7))
    pl.title('Barabasi & Albert graph: 400 Simulations,'+ dist +' distribution of initial attitude, 100-2000 nodes',fontsize=12)
    df['error'].hist(color='seagreen',bins=max(df['error']),normed=True,xlabelsize =10)
    pl.ylabel('Proportion')
    pl.xlabel('Error between actual source and inferred source (number of hops)')
    pl.savefig("ba_%s.png" % dist,dpi=500) 
    pl.show()

#Watts-Strogatz plots
x=sw_sim.loc[ba_sim['Initial_attitude_dist'] != 'x']['Delta']
df=x.to_frame(name='error')
axis=(max(df['error']))+2
pl.figure(figsize=(10,7))
pl.title('Watts-Strogatz graph: 1600 Simulations, 100-2000 nodes',fontsize=12)
df['error'].hist(color='seagreen',bins=np.arange(axis),normed=True,xlabelsize =10)
pl.ylabel('Proportion')
pl.xlabel('Error between actual source and inferred source (number of hops)')
pl.savefig("sw_ALL.png",dpi=500) 
pl.show()


#Watts-Strogatz plots
dists = ['normal', 'beta', 'exp1', 'uniform']
# Loop across all 4 distributions
for dist in dists:  
    x=sw_sim.loc[sw_sim['Initial_attitude_dist'] == dist]['Delta']
    df=x.to_frame(name='error')
    pl.figure(figsize=(10,7))
    pl.title('Watts-Strogatz graph: 400 Simulations, '+ dist+' distribution of initial attitude, 100-2000 nodes',fontsize=12)
    df['error'].hist(color='seagreen',bins=max(df['error']),normed=True,xlabelsize =10)
    pl.ylabel('Proportion')
    pl.xlabel('Error between actual source and inferred source (number of hops)')
    pl.savefig("sw_%s.png" % dist,dpi=500) 
    pl.show()

    
# FB graph 
# Initial attitude distributions
dists = ['normal', 'beta', 'exp1', 'uniform']
# Loop across all 4 distributions
for dist in dists:  

    x=fb_stats_all.loc[fb_stats_all['Initial_attitude_dist'] == dist]['Delta']
    df=x.to_frame(name='error')
    fig, ax = pl.subplots(figsize=(10,7)) 
    xticks = [0,1,2,3,4,5]
    ax.xaxis.set_ticks(xticks)
    ax.set_xticklabels(xticks, fontsize=16)
    axis=(max(df['error']))+2
    pl.title('Facebook graph: 100 Simulations, '+ dist +' distribution of initial attitude, 400 nodes',fontsize=12)
    df['error'].hist(color='seagreen',bins= np.arange(axis),normed=True,xlabelsize =10)
    pl.ylabel('Proportion')
    pl.xlabel('Error between actual source and inferred source (number of hops)')
    pl.savefig("fb_%s.png" % dist,dpi=500) 
    pl.show()
    
    
    
dists = ['normal', 'beta', 'exp1', 'uniform']
# Loop across all 4 distributions
for dist in dists:  

    x=fb_stats_all.loc[fb_stats_all['Initial_attitude_dist'] != 'x']['Delta']
    df=x.to_frame(name='error')
    fig, ax = pl.subplots(figsize=(10,7)) 
    xticks = [0,1,2,3,4,5]
    ax.xaxis.set_ticks(xticks)
    ax.set_xticklabels(xticks, fontsize=16)
    axis=(max(df['error']))+2
    pl.title('Facebook graph: 400 Simulations, 500 nodes',fontsize=12)
    df['error'].hist(color='seagreen',bins= np.arange(axis),normed=True,xlabelsize =10)
    pl.ylabel('Proportion')
    pl.xlabel('Error between actual source and inferred source (number of hops)')
    pl.savefig("fb_ALL.png",dpi=500) 
    pl.show()
    
    
    

"""
Scaling behaviour of source inference algorithm 

"""
#SW
dat=sw_sim.groupby(['Nodes'])['SI_time'].mean()
dat1=ba_sim.groupby(['Nodes'])['SI_time'].mean()
x=dat.rename("Watts Strogatz (density @1000 nodes ~0.002)")
y=dat1.rename("Barabasi Albert (density @1000 nodes ~0.002)")
x.plot(kind='line',figsize=(13,7),legend=True)
y.plot(kind='line',figsize=(13,7),legend=True)
pl.title('Source inference algorithm scaling, Barabasi & Albert vs Watts-Strogatz graphs: 1600 Simultions',fontsize=12)
pl.ylabel('Time (seconds)')
pl.xlabel('Graph size (nodes)')
pl.savefig("scaling.png",dpi=1000) 
pl.show()


dists = ['normal', 'beta', 'exp1', 'uniform']
# Loop across all 4 distributions
for dist in dists:  

    x=sw_sim.loc[sw_sim['Initial_attitude_dist'] == dist]
    dat=x.groupby(['Nodes'])['SI_time'].mean()
    pl.title('Facebook graph: 100 Simultions, '+ dist +' distribution of initial attitude, 400 nodes',fontsize=12)
    dat.plot(color='seagreen',figsize=(13,7))
    pl.ylabel('Proportion %')
    pl.xlabel('Error between actual source and inferred source (number of hops)')
    pl.show()
    
    
    
dat=sw_sim.groupby(['Time'])['Delta'].mean()
dat1=ba_sim.groupby(['Time'])['Delta'].mean()
x=dat.rename("Watts Strogatz (density @1000 nodes ~0.002)")
y=dat1.rename("Barabasi Albert (density @1000 nodes ~0.002)")
x.plot(kind='line',figsize=(13,7),legend=True)
y.plot(kind='line',figsize=(13,7),legend=True)
pl.title('Source inference algorithm scaling, Barabasi & Albert vs Watts-Strogatz graphs: 1600 Simultions',fontsize=12)
pl.ylabel('Time (seconds)')
pl.xlabel('Graph size (nodes)')
pl.savefig("scaling.png",dpi=1000) 
pl.show()
    
    
# Proportion of graph to observe rumour vs error 
    
dat=sw_sim.groupby(['Delta'])['Rumour_Prop'].mean()
dat1=ba_sim.groupby(['Delta'])['Rumour_Prop'].mean()
dat2=fb_stats_all.groupby(['Delta'])['Rumour_Prop'].mean()

x=dat.rename("Watts Strogatz (density @1000 nodes ~0.002)")
y=dat1.rename("Barabasi Albert (density @1000 nodes ~0.002)")
z=dat2.rename("Facebook (density @1000 nodes ~0.002)")
z.plot(kind='line',figsize=(13,7),legend=True)
y.plot(kind='line',figsize=(13,7),legend=True)
x.plot(kind='line',figsize=(13,7),legend=True)
pl.title('Relationship between proportion of graph to observe rumour and source inference error',fontsize=12)
pl.ylabel('Proportion of graph to observe rumour')
pl.xlabel('Mean error between actual source and inferred source (number of hops)')
#pl.savefig("error_prop.png",dpi=1000) 
pl.show()
    

dists = ['normal', 'beta', 'exp1', 'uniform']
# Loop across all 4 distributions
for dist in dists:  
    
    a=sw_sim.loc[sw_sim['Initial_attitude_dist'] == dist]
    b=ba_sim.loc[ba_sim['Initial_attitude_dist'] == dist]
    c=fb_stats_all.loc[fb_stats_all['Initial_attitude_dist'] == dist]

    dat=a.groupby(['Delta'])['Rumour_Prop'].mean()
    dat1=b.groupby(['Delta'])['Rumour_Prop'].mean()
    dat2=c.groupby(['Delta'])['Rumour_Prop'].mean()
    
    x=dat.rename("Watts Strogatz (400 Simulations, 100-2000 nodes)")
    y=dat1.rename("Barabasi Albert (400 Simulations, 100-2000 nodes)")
    z=dat2.rename("Facebook (100 Simulations, 500 nodes)")
    z.plot(kind='line',figsize=(13,7),legend=True)
    y.plot(kind='line',figsize=(13,7),legend=True)
    x.plot(kind='line',figsize=(13,7),legend=True)
    pl.title('Proportion of graph to observe rumour vs source inference error ('+dist+' Initial attitude)',fontsize=12)
    pl.ylabel('Proportion of graph to observe rumour')
    pl.xlabel('Mean error between actual source and inferred source (number of hops)')
    pl.savefig("error_prop_%s.png" % dist,dpi=1000) 
    pl.show()
        

    
 #graph Size vs error    
dists = ['normal', 'beta', 'exp1', 'uniform']
# Loop across all 4 distributions
for dist in dists:  
    
    a=sw_sim.loc[sw_sim['Initial_attitude_dist'] == dist]
    b=ba_sim.loc[ba_sim['Initial_attitude_dist'] == dist]
    c=fb_stats_all.loc[fb_stats_all['Initial_attitude_dist'] == dist]

    dat=a.groupby(['Nodes'])['Delta'].mean()
    dat1=b.groupby(['Nodes'])['Delta'].mean()

    
    x=dat.rename("Watts Strogatz (400 Simulations, 100-2000 nodes)")
    y=dat1.rename("Barabasi Albert (400 Simulations, 100-2000 nodes)")


    y.plot(kind='line',figsize=(13,7),legend=True)
    x.plot(kind='line',figsize=(13,7),legend=True)
    pl.title('Graph size vs source inference error ('+dist+' Initial attitude)',fontsize=12)
    pl.ylabel('Graph size (number of nodes')
    pl.ylabel('Mean error between actual source and inferred source (number of hops)')
    pl.savefig("error_size_%s.png" % dist,dpi=1000) 
    pl.show()
        

dat=sw_sim.groupby(['Time'])['Delta'].mean()
dat1=ba_sim.groupby(['Time'])['Delta'].mean()
x=dat.rename("Watts Strogatz (density @1000 nodes ~0.002)")
y=dat1.rename("Barabasi Albert (density @1000 nodes ~0.002)")
x.plot(kind='line',figsize=(13,7),legend=True)
y.plot(kind='line',figsize=(13,7),legend=True)
pl.title('Source inference algorithm scaling, Barabasi & Albert vs Watts-Strogatz graphs: 1600 Simultions',fontsize=12)
pl.ylabel('Time (seconds)')
pl.xlabel('Graph size (nodes)')
pl.savefig("scaling.png",dpi=1000) 
pl.show()
    
#time taken for rumour to converge/stop spreading 

dat=ba_sim.groupby(['Nodes'])['Time'].mean()
df=dat.to_frame()
df.plot(color='seagreen',figsize=(13,7))

dat=sw_sim.groupby(['Nodes'])['Time'].mean()
df=dat.to_frame()
df.plot(color='seagreen',figsize=(13,7))

#Rumour proportion heard
dat=ba_sim.groupby(['Nodes'])['Rumour_Prop'].mean()
df=dat.to_frame()
df.plot(color='seagreen',figsize=(13,7))

dat=sw_sim.groupby(['Nodes'])['Rumour_Prop'].mean()
df=dat.to_frame()
df.plot(color='seagreen',figsize=(13,7))



# Time vs rumour prop
dat=ba_sim.groupby(['Time'])['Rumour_Prop'].mean()
df=dat.to_frame()
df.plot(color='seagreen',figsize=(13,7))


x=ba_sim.loc[sw_sim['Initial_attitude_dist'] == 'beta']
dat=x.groupby(['Time'])['Rumour_Prop'].mean()
df=dat.to_frame()
df.plot(color='seagreen',figsize=(13,7))

x=ba_sim.loc[sw_sim['Initial_attitude_dist'] == 'exp1']
dat=x.groupby(['Time'])['Rumour_Prop'].mean()
df=dat.to_frame()
df.plot(color='seagreen',figsize=(13,7))

x=ba_sim.loc[sw_sim['Initial_attitude_dist'] == 'normal']
dat=x.groupby(['Time'])['Rumour_Prop'].mean()
df=dat.to_frame()
df.plot(color='seagreen',figsize=(13,7))

x=ba_sim.loc[sw_sim['Initial_attitude_dist'] == 'uniform']
dat=x.groupby(['Time'])['Rumour_Prop'].mean()
df=dat.to_frame()
df.plot(color='seagreen',figsize=(13,7))

dat=sw_sim.groupby(['Time'])['Rumour_Prop'].mean()
df=dat.to_frame()
df.plot(color='seagreen',figsize=(13,7))



# The end - thanks!






