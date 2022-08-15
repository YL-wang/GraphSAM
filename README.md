# GraphSAM
## Experiments


### Training
GraphSAINT <br>
``
python saint_graph.py --epochs <epochs> --load_CL <load_CL> --par <par> --rate <rate> -topk <topk>
``
<br>
where `` <par> `` is a contrastive loss ratio. `` <rate> `` is the perturbation ratio of data augmentation. 
`` <topk> `` is the number of subgraphs involved in contrastive learning. `` <load_CL> `` is to add contrastive learning at the Nth epoch, default is 0.

Cluster-GCN <br>
``
python cluster_graph.py --epochs <epochs> --load_CL <load_CL> --par <par> --rate <rate>
``
<br>

GraphSAGE <br>
``
python ns_graph.py --epochs <epochs> --par <par> --rate <rate>
``

```python
from GraphSAM import GraphSAM

model = YourModel()
criterion = YourCriterion()
base_optimizer = YourBaseOptimizer

optimizer = GraphSAM(
    params=model.parameters(),
    rho=0.05,
    beta=0.99,
    gamma=0.5,
    lambda=1,
    base_optimizer=base_optimizer,
    **kwargs
)
for epoch in range(epochs):
	i=0
	for batch in (train_loader):
		def closure():
			output = model(input)
			loss = loss_f(output, target)
			loss.backward()
			return loss

		if i==0:
			output = model(input)
			loss=loss_f(output,target)
			optimizer.step(i, epoch, closure, loss)
		else:
			optimizer.step(i, epoch, closure)
		loss = optimizer.get_loss()
		optimizer.zero_grad()
...

    ```
