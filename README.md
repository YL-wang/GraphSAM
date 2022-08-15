# GraphSAM
## Experiments


### Training
cd CoMPT-main
``
python train_graph.py --seed 777 --fold 5 --epochs 30 --dataset <datasets> --split random --sam <GraphSAM> --rho <rho> --radius <radius> --epoch_steps <lambda> --alpha <beta> --gamma <gamma>
``
<br>
where `` <par> `` is a contrastive loss ratio. `` <rate> `` is the perturbation ratio of data augmentation. 
`` <topk> `` is the number of subgraphs involved in contrastive learning. `` <load_CL> `` is to add contrastive learning at the Nth epoch, default is 0.
<br>
###GROVER <br>
cd grover-main
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
