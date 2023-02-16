# Import all the modules
import numpy as np
from pykeen.pipeline import pipeline, plot_losses, plot, plot_er
from pykeen.models import predict
from pykeen.triples import TriplesFactory
from matplotlib import pyplot as plt
from typing import List
import pykeen.nn
import torch
import os.path

# Save the latent vectors of entities and relations
def save_vectors(entity_representation_tensor,relation_embedding_tensor,m):
    entity_representation_tensor = entity_representation_tensor.detach().numpy()
    relation_embedding_tensor = relation_embedding_tensor.detach().numpy()
    if not os.path.exists('embed/french/'+str(m)+'/vectors'):
        os.makedirs('embed/french/'+str(m)+'/vectors')
    np.savetxt("embed/french/" + str(m) +"/vectors/entity_representation.txt", entity_representation_tensor, delimiter="\t",fmt='%s')
    np.savetxt("embed/french/" + str(m) +"/vectors/relation_representation.txt", relation_embedding_tensor, delimiter="\t",fmt='%s')

# Load the input KG
def load_dataset(name):
    triple_data = open(name).read().strip()
    data = np.array([triple.split('\t') for triple in triple_data.split('\n')])
    tf_data = TriplesFactory.from_labeled_triples(triples=data)
    entity_label =tf_data.entity_to_id.keys()
    relation_label = tf_data.relation_to_id.keys()
    return tf_data, triple_data, entity_label, relation_label

# Define the model with required arguments for training
def create_model(tf_training, tf_testing, embedding, n_epoch, path):
    results = pipeline(
        training=tf_training,
        testing=tf_testing,
        model=embedding,
        training_loop='sLCWA',
        model_kwargs=dict(embedding_dim=50),
        # Training configuration
        training_kwargs=dict(
            num_epochs=n_epoch,
            use_tqdm_batch=False,
        ),
        # Runtime configuration
        random_seed=1235,
        device='cpu',
    )
    model = results.model
    results.save_to_directory(path + embedding)
    return model, results

# plot the loss per epoch while training
def plotting(results,m):
        plot_losses(results)
        plt.savefig("embed/french/" + m + "/loss_plot.png", dpi=300)

# predict all the triples
def prediction(model, training, label):
    pred = predict.get_all_prediction_df(model=model, triples_factory=training)
    pred.to_csv('embed/french/' + label + '/pred_all_triples.csv', index=False)

# to store the entity and relation representation
def get_learned_embeddings(model):
    entity_representation_modules: List['pykeen.nn.RepresentationModule'] = model.entity_representations
    relation_representation_modules: List['pykeen.nn.RepresentationModule'] = model.relation_representations

    entity_embeddings: pykeen.nn.Embedding = entity_representation_modules[0]
    relation_embeddings: pykeen.nn.Embedding = relation_representation_modules[0]

    entity_embedding_tensor: torch.FloatTensor = entity_embeddings()
    relation_embedding_tensor: torch.FloatTensor = relation_embeddings()
    return entity_embedding_tensor, relation_embedding_tensor

# to predict the tail entity
def tail_prediction(model, head, relation, training, label, entity_label):
    pred = predict.get_prediction_df(model=model, head_label=head, relation_label=relation, triples_factory=training)
    pred.to_csv('embed/french/' + label + '/pred_tails.csv', index=False)

# to predict the head entity
def head_prediction(model, relation, tail, training, label, entity_label):
            pred_heads = predict.get_prediction_df(model=model, relation_label= relation, tail_label= tail, triples_factory=training)
            pred_heads.to_csv('embed/french/'+label +'/pred_heads.csv', index=False)

if __name__ == "__main__":
    # Input KG
    tf, triple_data, entity_label, relation_label = load_dataset('frenchEncoded.nt')
    # Split them into train, test
    training, testing = tf.split(random_state=1234)

    # ## Defining models which InterpretME will currently using for embeddings. If models like ConvE or ComplEx, please define the bacth size in create model()
    models = 'TransE'
    model, results = create_model(tf_training=training,tf_testing=testing, embedding=models, n_epoch=200, path='embed/french/')
    plotting(results,models)
    entity_representation_tensor, relation_embedding_tensor = get_learned_embeddings(model)
    tail_prediction(model =model, head='<http://dbpedia.org/resource/Charles_the_Simple>', relation='<http://dbpedia.org/ontology/hasSpouse>',training = results.training, label=models, entity_label= entity_label)