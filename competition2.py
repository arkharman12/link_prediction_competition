#!/usr/bin/env python
# coding: utf-8

# In[2]:


# useful libraries
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import csv


# In[3]:


# read in the competition graph
G = nx.Graph()

with open('train_edges.csv') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
        if row[1]=='1':
            edge= row[0].split('-')
            G.add_edge(int(edge[0]),int(edge[1]))


# In[4]:


# how many nodes in the network?
G.number_of_nodes()


# In[5]:


# how many edges in the network?
G.number_of_edges()


# In[6]:


# draw the network
nx.draw(G,node_size=1)


# In[7]:


# read in possible edges from the test file
Gsub = nx.Graph()

with open('sample_submission.csv') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
        if row[1]=='1' or row[1]=='0':
            edge= row[0].split('-')
            Gsub.add_edge(int(edge[0]),int(edge[1]))


# In[8]:


# how many nodes in the network?           
Gsub.number_of_nodes()


# In[9]:


# how many edges in the network?           
Gsub.number_of_edges()


# In[10]:


# add the nodes from the test data to training network
G.add_nodes_from(Gsub.nodes)


# In[11]:


# read in the features
features = np.genfromtxt('features.csv', delimiter=',')


# In[12]:


print(features.shape)
print(features[0,:])


# In[13]:


# dot product of features
y1=features[features[:,0]==462][0][1:1433]
y2=features[features[:,0]==1175][0][1:1433]

print(np.dot(y1,y2))


# In[14]:


from node2vec import Node2Vec

# pre-compute the probabilities and and generate walks
node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=10, workers=1)


# In[15]:


# embed the nodes
model = node2vec.fit(window=10, min_count=1, batch_words=4)


# In[16]:


from node2vec.edges import HadamardEmbedder

# embed the edges
edges_embs = HadamardEmbedder(keyed_vectors=model.wv)


# In[19]:


# NOTE: I have put all the different things I tried in this section of the code. Comment it out for readability
# Of course if you tried to run everything together it's NOT going to work. Since I ran everything separately and got results
# But if you had like you can uncomment/comment each of them and get the prediction of valid/invalid edge in form of 1 or 0

with open('comp2_submission.csv', 'w') as csvfile:
    fieldnames = ['edge','label']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    with open('sample_submission.csv') as csvfile2:
        reader = csv.reader(csvfile2, delimiter=',')
        for row in reader:
            if row[1]=='1' or row[1]=='0':
                edge= row[0].split('-')

                i=int(edge[0])
                # print('i: %d' % (i))
                j=int(edge[1])
                # print('j: %d' % (j))
                
                ########## Features ##########
                y1=features[features[:,0]==i][0][1:1433]
                y2=features[features[:,0]==j][0][1:1433]
                
                ########## Dot Product of Features ##########
                dp=np.dot(y1,y2)
                # print('Dp: %d' % (dp))
                
                ########## Tried Node Embedding with Node2vec ##########
                model.wv.get_vector(str(edge[0]))
                model.wv.get_vector(str(edge[1]))
                
                ########## Tried Edge Embedding with Node2vec ##########
                edges_embs[(edge[0], edge[1])]
                edges_kv = edges_embs.as_keyed_vectors()
                edges_kv.most_similar(str((edge[0], edge[1])))
        
                ########## Tried to find most similar nodes and make prediction according to that ##########
                nodeP = model.wv.most_similar(str(edge[0]), str(edge[1])) # also works with str(row[1])
                # print(nodeP)
                for p in nodeP:
                    if p[1] > 0:
                        prediction=1
                    else:
                        prediction=0


                ########## Tried to find most similar nodes for just the left node and join it ##########
                ########## with the most similar nodes and make the prediction ##########
                nodeP = model.wv.most_similar(str(edge[0])) # For left node i
                # print(nodeP)
                for p in nodeP:
                    # print(j) # Right node j
                    # print(p[0]) # Most similar node to i
                    # print(p[1]) # Percentage of that node

                    if p[1] > 0:
                        # edg_out=str(i)+"-"+p[0]
                        prediction=1
                    else:
                        prediction=0
                        
                        
                ########## Tried to find most similar nodes for the right node this time and join it ##########
                ########## with the most similar nodes and make the prediction ##########
                nodeP = model.wv.most_similar(str(edge[1])) # For right node j
                # print(nodeP)
                for p in nodeP:
                    # print(i) # Left node i
                    # print(p[0]) # Most similar node to j

                    if p[1] > 0:
                        # edg_out=str(j)+"-"+p[0]
                        prediction=1
                    else:
                        prediction=0
                        
                
                ########## Tried bunch of Algorithms ##########
                
                ########## Common Neighbors ##########
                prediction1= len(sorted(nx.common_neighbors(G, i, j))) # do >=0
                # print('Model1 Pred:' + str(prediction1))
                
                
                ########## Resource Allocation Index ##########
                prediction2 = sorted(nx.resource_allocation_index(G, [(i, j)])) # do >0
                # print(prediction2)
                # print(prediction2[0][2])
                # prediction2 = prediction2[0][2]
                # print('Model2 Pred:' + str(prediction2))
                
                
                ########## Jaccard Coefficient ##########
                prediction3 = sorted(nx.jaccard_coefficient(G, [(i, j)])) # do >0
                # print(prediction3)
                # print(prediction3[0][2])
                # prediction3 = prediction3[0][2]
                # print('Model3 Pred:' + str(prediction3))
                
                
                ########## Adamic Adar Index ##########
                prediction4 = sorted(nx.adamic_adar_index(G, [(i, j)])) # do >0
                # print(prediction4)
                # print(prediction4[0][2])
                # prediction4 = prediction4[0][2]
                # print('Model4 Pred:' + str(prediction4))

                
                ########## Prederential Attachment ##########
                prediction5 = sorted(nx.preferential_attachment(G, [(i, j)])) # do >0
                # print(prediction5)
                # print(prediction5[0][2])
                # prediction5 = prediction5[0][2]
                # print('Model5 Pred:' + str(prediction5))

                
                ########## Tried couple different ways to get the score up ##########
                ########## Really thought something like this would work well ##########
                if prediction1 >0 and prediction2 >0 and prediction3 >0 and prediction4 >0 and prediction5 >0 and dp >=2:
                    prediction=1
                else:
                    prediction=0
                    
                ########## This is another one ##########
                if dp>2 and prediction[0][2]>0:
                    prediction=1
                else:
                    prediction=0
                
                ########## Print the prediction with left and right node ##########
                edg_out=str(i)+"-"+str(j)    
                writer.writerow({'edge': edg_out,'label':prediction})
                print(edg_out, prediction)

