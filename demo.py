# %%


from database import NeuronDB 

db = NeuronDB()

# %%

neuronpedia_request = {
    "model_id": "gemma-2-2b",
    "dictionaries": [
        {
            "layer_id": "0-gemmascope-res-16k",
            "indices": [0, 1, 2]
        }
    ]
}

db.cache_neuronpedia(neuronpedia_request)
