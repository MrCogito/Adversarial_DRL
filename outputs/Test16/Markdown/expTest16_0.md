Traceback (most recent call last):
  File "/zhome/59/9/198225/Desktop/Adversarial_DRL/Adversarial_DRL/main.py", line 2, in <module>
    from ppo_training import PPOAgent, train_ppo  # Import from ppo_combined.py
  File "/zhome/59/9/198225/Desktop/Adversarial_DRL/Adversarial_DRL/ppo_training.py", line 2, in <module>
    import torch
  File "/zhome/59/9/198225/Desktop/Adversarial_DRL/project-env/lib/python3.9/site-packages/torch/__init__.py", line 1382, in <module>
    from .functional import *  # noqa: F403
  File "/zhome/59/9/198225/Desktop/Adversarial_DRL/project-env/lib/python3.9/site-packages/torch/functional.py", line 7, in <module>
    import torch.nn.functional as F
  File "/zhome/59/9/198225/Desktop/Adversarial_DRL/project-env/lib/python3.9/site-packages/torch/nn/__init__.py", line 1, in <module>
    from .modules import *  # noqa: F403
  File "/zhome/59/9/198225/Desktop/Adversarial_DRL/project-env/lib/python3.9/site-packages/torch/nn/modules/__init__.py", line 10, in <module>
    from .loss import L1Loss, NLLLoss, KLDivLoss, MSELoss, BCELoss, BCEWithLogitsLoss, NLLLoss2d, \
  File "<frozen importlib._bootstrap>", line 1007, in _find_and_load
  File "<frozen importlib._bootstrap>", line 986, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 680, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 846, in exec_module
  File "<frozen importlib._bootstrap_external>", line 941, in get_code
  File "<frozen importlib._bootstrap_external>", line 1040, in get_data
KeyboardInterrupt

------------------------------------------------------------
Sender: LSF System <lsfadmin@hpc.dtu.dk>
Subject: Job 19760209: <Test16_0> in cluster <dcc> Exited

Job <Test16_0> was submitted from host <gbarlogin2> by user <s230208> in cluster <dcc> at Tue Dec 12 20:20:54 2023
Job was executed on host(s) <4*n-62-20-4>, in queue <gpuv100>, as user <s230208> in cluster <dcc> at Tue Dec 12 20:20:55 2023
</zhome/59/9/198225> was used as the home directory.
</zhome/59/9/198225/Desktop/Adversarial_DRL/Adversarial_DRL> was used as the working directory.
Started at Tue Dec 12 20:20:55 2023
Terminated at Tue Dec 12 20:21:35 2023
Results reported at Tue Dec 12 20:21:35 2023

Your job looked like:

------------------------------------------------------------
# LSBATCH: User input
#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[mem=16G]"
#BSUB -R "select[gpu16gb]"
#BSUB -R "span[hosts=1]"
#BSUB -W 1440
# end of BSUB options
module -s load python3
source ../project-env/bin/activate

python main.py $MYARGS
------------------------------------------------------------

TERM_OWNER: job killed by owner.
Exited with exit code 130.

Resource usage summary:

    CPU time :                                   1.77 sec.
    Max Memory :                                 169 MB
    Average Memory :                             142.00 MB
    Total Requested Memory :                     65536.00 MB
    Delta Memory :                               65367.00 MB
    Max Swap :                                   -
    Max Processes :                              4
    Max Threads :                                5
    Run time :                                   119 sec.
    Turnaround time :                            41 sec.

The output (if any) is above this job summary.

wandb: Currently logged in as: mrcogito (deeplearning_painn). Use `wandb login --relogin` to force relogin
wandb: wandb version 0.16.1 is available!  To upgrade, please run:
wandb:  $ pip install wandb --upgrade
wandb: Tracking run with wandb version 0.16.0
wandb: Run data is saved locally in /zhome/59/9/198225/Desktop/Adversarial_DRL/Adversarial_DRL/wandb/run-20231212_202253-liuzxvri
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run Test16-0
wandb: ⭐️ View project at https://wandb.ai/deeplearning_painn/adversarial_dlr
wandb: 🚀 View run at https://wandb.ai/deeplearning_painn/adversarial_dlr/runs/liuzxvri

<style>
c { color: #9cdcfe; font-family: 'Verdana', sans-serif;} /* VARIABLE */
d { color: #4EC9B0; font-family: 'Verdana', sans-serif;} /* CLASS */
e { color: #569cd6; font-family: 'Verdana', sans-serif;} /* BOOL */
f { color: #b5cea8; font-family: 'Verdana', sans-serif;} /* NUMBERS */
j { color: #ce9178; font-family: 'Verdana', sans-serif;} /* STRING */
k { font-family: 'Verdana', sans-serif;} /* SYMBOLS */
</style>

# Parameters

| PARAMETER         | TYPE              | VALUE             |
|-------------------|-------------------|-------------------|
| <c>name</c>       | <d>str</d>        | <j>"Test16-0"</j> |
| <c>time</c>       | <d>int</d>        | <f>84600</f>      |
| <c>epochs</c>     | <d>int</d>        | <f>1000</f>       |
| <c>batch_size</c> | <d>int</d>        | <f>32</f>         |
| <c>gamma</c>      | <d>float</d>      | <f>99.0</f>       |
| <c>train_agent</c>| <d>function</d>   | None              |

# Output

```
Starting training with PPO Agent
Starting Epoch 1/1000
Starting Epoch 2/1000
Starting Epoch 3/1000
Starting Epoch 4/1000
Starting Epoch 5/1000
Starting Epoch 6/1000
Starting Epoch 7/1000
Starting Epoch 8/1000
Starting Epoch 9/1000
Starting Epoch 10/1000
Starting Epoch 11/1000
Starting Epoch 12/1000
Starting Epoch 13/1000
Starting Epoch 14/1000
Starting Epoch 15/1000
Starting Epoch 16/1000
Starting Epoch 17/1000
Starting Epoch 18/1000
Starting Epoch 19/1000
Starting Epoch 20/1000
Starting Epoch 21/1000
Starting Epoch 22/1000
Starting Epoch 23/1000
Starting Epoch 24/1000
Starting Epoch 25/1000
Starting Epoch 26/1000
Starting Epoch 27/1000
Starting Epoch 28/1000
Starting Epoch 29/1000
Starting Epoch 30/1000
Starting Epoch 31/1000
Starting Epoch 32/1000
Starting Epoch 33/1000
Starting Epoch 34/1000
Starting Epoch 35/1000
Starting Epoch 36/1000
Starting Epoch 37/1000
Starting Epoch 38/1000
Starting Epoch 39/1000
Starting Epoch 40/1000
Starting Epoch 41/1000
Starting Epoch 42/1000
Starting Epoch 43/1000
Starting Epoch 44/1000
Starting Epoch 45/1000
Starting Epoch 46/1000
Starting Epoch 47/1000
Starting Epoch 48/1000
Starting Epoch 49/1000
Starting Epoch 50/1000
Starting Epoch 51/1000
Starting Epoch 52/1000
Starting Epoch 53/1000
Starting Epoch 54/1000
Starting Epoch 55/1000
Starting Epoch 56/1000
Starting Epoch 57/1000
Starting Epoch 58/1000
Starting Epoch 59/1000
Starting Epoch 60/1000
Starting Epoch 61/1000
Starting Epoch 62/1000
Starting Epoch 63/1000
Starting Epoch 64/1000
Starting Epoch 65/1000
Starting Epoch 66/1000
Starting Epoch 67/1000
Starting Epoch 68/1000
Starting Epoch 69/1000
Starting Epoch 70/1000
Starting Epoch 71/1000
Starting Epoch 72/1000
Starting Epoch 73/1000
Starting Epoch 74/1000
Starting Epoch 75/1000
Starting Epoch 76/1000
Starting Epoch 77/1000
Starting Epoch 78/1000
Starting Epoch 79/1000
Starting Epoch 80/1000
Starting Epoch 81/1000
Starting Epoch 82/1000
Starting Epoch 83/1000
Starting Epoch 84/1000
Starting Epoch 85/1000
Starting Epoch 86/1000
Starting Epoch 87/1000
Starting Epoch 88/1000
Starting Epoch 89/1000
Starting Epoch 90/1000
Starting Epoch 91/1000
Starting Epoch 92/1000
Starting Epoch 93/1000
Starting Epoch 94/1000
Starting Epoch 95/1000
Starting Epoch 96/1000
Starting Epoch 97/1000
Starting Epoch 98/1000
Starting Epoch 99/1000
Starting Epoch 100/1000
Starting Epoch 101/1000
Starting Epoch 102/1000
Starting Epoch 103/1000
Starting Epoch 104/1000
Starting Epoch 105/1000
Starting Epoch 106/1000
Starting Epoch 107/1000
Starting Epoch 108/1000
Starting Epoch 109/1000
Starting Epoch 110/1000
Starting Epoch 111/1000
Starting Epoch 112/1000
Starting Epoch 113/1000
Starting Epoch 114/1000
Starting Epoch 115/1000
Starting Epoch 116/1000
Starting Epoch 117/1000
Starting Epoch 118/1000
Starting Epoch 119/1000
Starting Epoch 120/1000
Starting Epoch 121/1000
Starting Epoch 122/1000
Starting Epoch 123/1000
Starting Epoch 124/1000
Starting Epoch 125/1000
Starting Epoch 126/1000
Starting Epoch 127/1000
Starting Epoch 128/1000
Starting Epoch 129/1000
Starting Epoch 130/1000
Starting Epoch 131/1000
Starting Epoch 132/1000
Starting Epoch 133/1000
Starting Epoch 134/1000
Starting Epoch 135/1000
Starting Epoch 136/1000
Starting Epoch 137/1000
Starting Epoch 138/1000
Starting Epoch 139/1000
Starting Epoch 140/1000
Starting Epoch 141/1000
Starting Epoch 142/1000
Starting Epoch 143/1000
Starting Epoch 144/1000
Starting Epoch 145/1000
Starting Epoch 146/1000
Starting Epoch 147/1000
Starting Epoch 148/1000
Starting Epoch 149/1000
Starting Epoch 150/1000
Starting Epoch 151/1000
Starting Epoch 152/1000
Starting Epoch 153/1000
Starting Epoch 154/1000
Starting Epoch 155/1000
Starting Epoch 156/1000
Starting Epoch 157/1000
Starting Epoch 158/1000
Starting Epoch 159/1000
Starting Epoch 160/1000
Starting Epoch 161/1000
Starting Epoch 162/1000
Starting Epoch 163/1000
Starting Epoch 164/1000
Starting Epoch 165/1000
Starting Epoch 166/1000
Starting Epoch 167/1000
Starting Epoch 168/1000
Starting Epoch 169/1000
Starting Epoch 170/1000
Starting Epoch 171/1000
Starting Epoch 172/1000
Starting Epoch 173/1000
Starting Epoch 174/1000
Starting Epoch 175/1000
Starting Epoch 176/1000
Starting Epoch 177/1000
Starting Epoch 178/1000
Starting Epoch 179/1000
Starting Epoch 180/1000
Starting Epoch 181/1000
Starting Epoch 182/1000
Starting Epoch 183/1000
Starting Epoch 184/1000
Starting Epoch 185/1000
Starting Epoch 186/1000
Starting Epoch 187/1000
Starting Epoch 188/1000
Starting Epoch 189/1000
Starting Epoch 190/1000
Starting Epoch 191/1000
Starting Epoch 192/1000
Starting Epoch 193/1000
Starting Epoch 194/1000
Starting Epoch 195/1000
Starting Epoch 196/1000
Starting Epoch 197/1000
Starting Epoch 198/1000
Starting Epoch 199/1000
Starting Epoch 200/1000
Starting Epoch 201/1000
Starting Epoch 202/1000
Starting Epoch 203/1000
Starting Epoch 204/1000
Starting Epoch 205/1000
Starting Epoch 206/1000
Starting Epoch 207/1000
Starting Epoch 208/1000
Starting Epoch 209/1000
Starting Epoch 210/1000
Starting Epoch 211/1000
Starting Epoch 212/1000
Starting Epoch 213/1000
Starting Epoch 214/1000
Starting Epoch 215/1000
Starting Epoch 216/1000
Starting Epoch 217/1000
Starting Epoch 218/1000
Starting Epoch 219/1000
Starting Epoch 220/1000
Starting Epoch 221/1000
Starting Epoch 222/1000
Starting Epoch 223/1000
Starting Epoch 224/1000
Starting Epoch 225/1000
Starting Epoch 226/1000
Starting Epoch 227/1000
Starting Epoch 228/1000
Starting Epoch 229/1000
Starting Epoch 230/1000
Starting Epoch 231/1000
Starting Epoch 232/1000
Starting Epoch 233/1000
Starting Epoch 234/1000
Starting Epoch 235/1000
Starting Epoch 236/1000
Starting Epoch 237/1000
Starting Epoch 238/1000
Starting Epoch 239/1000
Starting Epoch 240/1000
Starting Epoch 241/1000
Starting Epoch 242/1000
Starting Epoch 243/1000
Starting Epoch 244/1000
Starting Epoch 245/1000
Starting Epoch 246/1000
Starting Epoch 247/1000
Starting Epoch 248/1000
Starting Epoch 249/1000
Starting Epoch 250/1000
Starting Epoch 251/1000
Starting Epoch 252/1000
Starting Epoch 253/1000
Starting Epoch 254/1000
Starting Epoch 255/1000
Starting Epoch 256/1000
Starting Epoch 257/1000
Starting Epoch 258/1000
Starting Epoch 259/1000
Starting Epoch 260/1000
Starting Epoch 261/1000
Starting Epoch 262/1000
Starting Epoch 263/1000
Starting Epoch 264/1000
Starting Epoch 265/1000
Starting Epoch 266/1000
Starting Epoch 267/1000
Starting Epoch 268/1000
Starting Epoch 269/1000
Starting Epoch 270/1000
Starting Epoch 271/1000
Starting Epoch 272/1000
Starting Epoch 273/1000
Starting Epoch 274/1000
Starting Epoch 275/1000
Starting Epoch 276/1000
Starting Epoch 277/1000
Starting Epoch 278/1000
Starting Epoch 279/1000
Starting Epoch 280/1000
Starting Epoch 281/1000
Starting Epoch 282/1000
Starting Epoch 283/1000
Starting Epoch 284/1000
Starting Epoch 285/1000
Starting Epoch 286/1000
Starting Epoch 287/1000
Starting Epoch 288/1000
Starting Epoch 289/1000
Starting Epoch 290/1000
Starting Epoch 291/1000
Starting Epoch 292/1000
Starting Epoch 293/1000
Starting Epoch 294/1000
Starting Epoch 295/1000
Starting Epoch 296/1000
Starting Epoch 297/1000
Starting Epoch 298/1000
Starting Epoch 299/1000
Starting Epoch 300/1000
Starting Epoch 301/1000
Starting Epoch 302/1000
Starting Epoch 303/1000
Starting Epoch 304/1000
Starting Epoch 305/1000
Starting Epoch 306/1000
Starting Epoch 307/1000
Starting Epoch 308/1000
Starting Epoch 309/1000
Starting Epoch 310/1000
Starting Epoch 311/1000
Starting Epoch 312/1000
Starting Epoch 313/1000
Starting Epoch 314/1000
Starting Epoch 315/1000
Starting Epoch 316/1000
Starting Epoch 317/1000
Starting Epoch 318/1000
Starting Epoch 319/1000
Starting Epoch 320/1000
Starting Epoch 321/1000
Starting Epoch 322/1000
Starting Epoch 323/1000
Starting Epoch 324/1000
Starting Epoch 325/1000
Starting Epoch 326/1000
Starting Epoch 327/1000
Starting Epoch 328/1000
Starting Epoch 329/1000
Starting Epoch 330/1000
Starting Epoch 331/1000
Starting Epoch 332/1000
Starting Epoch 333/1000
Starting Epoch 334/1000
Starting Epoch 335/1000
Starting Epoch 336/1000
Starting Epoch 337/1000
Starting Epoch 338/1000
Starting Epoch 339/1000
Starting Epoch 340/1000
Starting Epoch 341/1000
Starting Epoch 342/1000
Starting Epoch 343/1000
Starting Epoch 344/1000
Starting Epoch 345/1000
Starting Epoch 346/1000
Starting Epoch 347/1000
Starting Epoch 348/1000
Starting Epoch 349/1000
Starting Epoch 350/1000
Starting Epoch 351/1000
Starting Epoch 352/1000
Starting Epoch 353/1000
Starting Epoch 354/1000
Starting Epoch 355/1000
Starting Epoch 356/1000
Starting Epoch 357/1000
Starting Epoch 358/1000
Starting Epoch 359/1000
Starting Epoch 360/1000
Starting Epoch 361/1000
Starting Epoch 362/1000
Starting Epoch 363/1000
Starting Epoch 364/1000
Starting Epoch 365/1000
Starting Epoch 366/1000
Starting Epoch 367/1000
Starting Epoch 368/1000
Starting Epoch 369/1000
Starting Epoch 370/1000
Starting Epoch 371/1000
Starting Epoch 372/1000
Starting Epoch 373/1000
Starting Epoch 374/1000
Starting Epoch 375/1000
Starting Epoch 376/1000
Starting Epoch 377/1000
Starting Epoch 378/1000
Starting Epoch 379/1000
Starting Epoch 380/1000
Starting Epoch 381/1000
Starting Epoch 382/1000
Starting Epoch 383/1000
Starting Epoch 384/1000
Starting Epoch 385/1000
Starting Epoch 386/1000
Starting Epoch 387/1000
Starting Epoch 388/1000
Starting Epoch 389/1000
Starting Epoch 390/1000
Starting Epoch 391/1000
Starting Epoch 392/1000
Starting Epoch 393/1000
Starting Epoch 394/1000
Starting Epoch 395/1000
Starting Epoch 396/1000
Starting Epoch 397/1000
Starting Epoch 398/1000
Starting Epoch 399/1000
Starting Epoch 400/1000
Starting Epoch 401/1000
Starting Epoch 402/1000
Starting Epoch 403/1000
Starting Epoch 404/1000
Starting Epoch 405/1000
Starting Epoch 406/1000
Starting Epoch 407/1000
Starting Epoch 408/1000
Starting Epoch 409/1000
Starting Epoch 410/1000
Starting Epoch 411/1000
Starting Epoch 412/1000
Starting Epoch 413/1000
Starting Epoch 414/1000
Starting Epoch 415/1000
Starting Epoch 416/1000
Starting Epoch 417/1000
Starting Epoch 418/1000
Starting Epoch 419/1000
Starting Epoch 420/1000
Starting Epoch 421/1000
Starting Epoch 422/1000
Starting Epoch 423/1000
Starting Epoch 424/1000
Starting Epoch 425/1000
Starting Epoch 426/1000
Starting Epoch 427/1000
Starting Epoch 428/1000
Starting Epoch 429/1000
Starting Epoch 430/1000
Starting Epoch 431/1000
Starting Epoch 432/1000
Starting Epoch 433/1000
Starting Epoch 434/1000
Starting Epoch 435/1000
Starting Epoch 436/1000
Starting Epoch 437/1000
Starting Epoch 438/1000
Starting Epoch 439/1000
Starting Epoch 440/1000
Starting Epoch 441/1000
Starting Epoch 442/1000
Starting Epoch 443/1000
Starting Epoch 444/1000
Starting Epoch 445/1000
Starting Epoch 446/1000
Starting Epoch 447/1000
Starting Epoch 448/1000
Starting Epoch 449/1000
Starting Epoch 450/1000
Starting Epoch 451/1000
Starting Epoch 452/1000
Starting Epoch 453/1000
Starting Epoch 454/1000
Starting Epoch 455/1000
Starting Epoch 456/1000
Starting Epoch 457/1000
Starting Epoch 458/1000
Starting Epoch 459/1000
Starting Epoch 460/1000
Starting Epoch 461/1000
Starting Epoch 462/1000
Starting Epoch 463/1000
Starting Epoch 464/1000
Starting Epoch 465/1000
Starting Epoch 466/1000
Starting Epoch 467/1000
Starting Epoch 468/1000
Starting Epoch 469/1000
Starting Epoch 470/1000
Starting Epoch 471/1000
Starting Epoch 472/1000
Starting Epoch 473/1000
Starting Epoch 474/1000
Starting Epoch 475/1000
Starting Epoch 476/1000
Starting Epoch 477/1000
Starting Epoch 478/1000
Starting Epoch 479/1000
Starting Epoch 480/1000
Starting Epoch 481/1000
Starting Epoch 482/1000
Starting Epoch 483/1000
Starting Epoch 484/1000
Starting Epoch 485/1000
Starting Epoch 486/1000
Starting Epoch 487/1000
Starting Epoch 488/1000
Starting Epoch 489/1000
Starting Epoch 490/1000
Starting Epoch 491/1000
Starting Epoch 492/1000
Starting Epoch 493/1000
Starting Epoch 494/1000
Starting Epoch 495/1000
Starting Epoch 496/1000
Starting Epoch 497/1000
Starting Epoch 498/1000
Starting Epoch 499/1000
Starting Epoch 500/1000
Starting Epoch 501/1000
Starting Epoch 502/1000
Starting Epoch 503/1000
Starting Epoch 504/1000
Starting Epoch 505/1000
Starting Epoch 506/1000
Starting Epoch 507/1000
Starting Epoch 508/1000
Starting Epoch 509/1000
Starting Epoch 510/1000
Starting Epoch 511/1000
Starting Epoch 512/1000
Starting Epoch 513/1000
Starting Epoch 514/1000
Starting Epoch 515/1000
Starting Epoch 516/1000
Starting Epoch 517/1000
Starting Epoch 518/1000
Starting Epoch 519/1000
Starting Epoch 520/1000
Starting Epoch 521/1000
Starting Epoch 522/1000
Starting Epoch 523/1000
Starting Epoch 524/1000
Starting Epoch 525/1000
Starting Epoch 526/1000
Starting Epoch 527/1000
Starting Epoch 528/1000
Starting Epoch 529/1000
Starting Epoch 530/1000
Starting Epoch 531/1000
Starting Epoch 532/1000
Starting Epoch 533/1000
Starting Epoch 534/1000
Starting Epoch 535/1000
Starting Epoch 536/1000
Starting Epoch 537/1000
Starting Epoch 538/1000
Starting Epoch 539/1000
Starting Epoch 540/1000
Starting Epoch 541/1000
Starting Epoch 542/1000
Starting Epoch 543/1000
Starting Epoch 544/1000
Starting Epoch 545/1000
Starting Epoch 546/1000
Starting Epoch 547/1000
Starting Epoch 548/1000
Starting Epoch 549/1000
Starting Epoch 550/1000
Starting Epoch 551/1000
Starting Epoch 552/1000
Starting Epoch 553/1000
Starting Epoch 554/1000
Starting Epoch 555/1000
Starting Epoch 556/1000
Starting Epoch 557/1000
Starting Epoch 558/1000
Starting Epoch 559/1000
Starting Epoch 560/1000
Starting Epoch 561/1000
Starting Epoch 562/1000
Starting Epoch 563/1000
Starting Epoch 564/1000
Starting Epoch 565/1000
Starting Epoch 566/1000
Starting Epoch 567/1000
Starting Epoch 568/1000
Starting Epoch 569/1000
Starting Epoch 570/1000
Starting Epoch 571/1000
Starting Epoch 572/1000
Starting Epoch 573/1000
Starting Epoch 574/1000
Starting Epoch 575/1000
Starting Epoch 576/1000
Starting Epoch 577/1000
Starting Epoch 578/1000
Starting Epoch 579/1000
Starting Epoch 580/1000
Starting Epoch 581/1000
Starting Epoch 582/1000
Starting Epoch 583/1000
Starting Epoch 584/1000
Starting Epoch 585/1000
Starting Epoch 586/1000
Starting Epoch 587/1000
Starting Epoch 588/1000
Starting Epoch 589/1000
Starting Epoch 590/1000
Starting Epoch 591/1000
Starting Epoch 592/1000
Starting Epoch 593/1000
Starting Epoch 594/1000
Starting Epoch 595/1000
Starting Epoch 596/1000
Starting Epoch 597/1000
Starting Epoch 598/1000
Starting Epoch 599/1000
Starting Epoch 600/1000
Starting Epoch 601/1000
Starting Epoch 602/1000
Starting Epoch 603/1000
Starting Epoch 604/1000
Starting Epoch 605/1000
Starting Epoch 606/1000
Starting Epoch 607/1000
Starting Epoch 608/1000
Starting Epoch 609/1000
Starting Epoch 610/1000
Starting Epoch 611/1000
Starting Epoch 612/1000
Starting Epoch 613/1000
Starting Epoch 614/1000
Starting Epoch 615/1000
Starting Epoch 616/1000
Starting Epoch 617/1000
Starting Epoch 618/1000
Starting Epoch 619/1000
Starting Epoch 620/1000
Starting Epoch 621/1000
Starting Epoch 622/1000
Starting Epoch 623/1000
Starting Epoch 624/1000
Starting Epoch 625/1000
Starting Epoch 626/1000
Starting Epoch 627/1000
Starting Epoch 628/1000
Starting Epoch 629/1000
Starting Epoch 630/1000
Starting Epoch 631/1000
Starting Epoch 632/1000
Starting Epoch 633/1000
Starting Epoch 634/1000
Starting Epoch 635/1000
Starting Epoch 636/1000
Starting Epoch 637/1000
Starting Epoch 638/1000
Starting Epoch 639/1000
Starting Epoch 640/1000
Starting Epoch 641/1000
Starting Epoch 642/1000
Starting Epoch 643/1000
Starting Epoch 644/1000
Starting Epoch 645/1000
Starting Epoch 646/1000
Starting Epoch 647/1000
Starting Epoch 648/1000
Starting Epoch 649/1000
Starting Epoch 650/1000
Starting Epoch 651/1000
Starting Epoch 652/1000
Starting Epoch 653/1000
Starting Epoch 654/1000
Starting Epoch 655/1000
Starting Epoch 656/1000
Starting Epoch 657/1000
Starting Epoch 658/1000
Starting Epoch 659/1000
Starting Epoch 660/1000
Starting Epoch 661/1000
Starting Epoch 662/1000
Starting Epoch 663/1000
Starting Epoch 664/1000
Starting Epoch 665/1000
Starting Epoch 666/1000
Starting Epoch 667/1000
Starting Epoch 668/1000
Starting Epoch 669/1000
Starting Epoch 670/1000
Starting Epoch 671/1000
Starting Epoch 672/1000
Starting Epoch 673/1000
Starting Epoch 674/1000
Starting Epoch 675/1000
Starting Epoch 676/1000
Starting Epoch 677/1000
Starting Epoch 678/1000
Starting Epoch 679/1000
Starting Epoch 680/1000
Starting Epoch 681/1000
Starting Epoch 682/1000
Starting Epoch 683/1000
Starting Epoch 684/1000
Starting Epoch 685/1000
Starting Epoch 686/1000
Starting Epoch 687/1000
Starting Epoch 688/1000
Starting Epoch 689/1000
Starting Epoch 690/1000
Starting Epoch 691/1000
Starting Epoch 692/1000
Starting Epoch 693/1000
Starting Epoch 694/1000
Starting Epoch 695/1000
Starting Epoch 696/1000
Starting Epoch 697/1000
Starting Epoch 698/1000
Starting Epoch 699/1000
Starting Epoch 700/1000
Starting Epoch 701/1000
Starting Epoch 702/1000
Starting Epoch 703/1000
Starting Epoch 704/1000
Starting Epoch 705/1000
Starting Epoch 706/1000
Starting Epoch 707/1000
Starting Epoch 708/1000
Starting Epoch 709/1000
Starting Epoch 710/1000
Starting Epoch 711/1000
Starting Epoch 712/1000
Starting Epoch 713/1000
Starting Epoch 714/1000
Starting Epoch 715/1000
Starting Epoch 716/1000
Starting Epoch 717/1000
Starting Epoch 718/1000
Starting Epoch 719/1000
Starting Epoch 720/1000
Starting Epoch 721/1000
Starting Epoch 722/1000
Starting Epoch 723/1000
Starting Epoch 724/1000
Starting Epoch 725/1000
Starting Epoch 726/1000
Starting Epoch 727/1000
Starting Epoch 728/1000
Starting Epoch 729/1000
Starting Epoch 730/1000
Starting Epoch 731/1000
Starting Epoch 732/1000
Starting Epoch 733/1000
Starting Epoch 734/1000
Starting Epoch 735/1000
Starting Epoch 736/1000
Starting Epoch 737/1000
Starting Epoch 738/1000
Starting Epoch 739/1000
Starting Epoch 740/1000
Starting Epoch 741/1000
Starting Epoch 742/1000
Starting Epoch 743/1000
Starting Epoch 744/1000
Starting Epoch 745/1000
Starting Epoch 746/1000
Starting Epoch 747/1000
Starting Epoch 748/1000
Starting Epoch 749/1000
Starting Epoch 750/1000
Starting Epoch 751/1000
Starting Epoch 752/1000
Starting Epoch 753/1000
Starting Epoch 754/1000
Starting Epoch 755/1000
Starting Epoch 756/1000
Starting Epoch 757/1000
Starting Epoch 758/1000
Starting Epoch 759/1000
Starting Epoch 760/1000
Starting Epoch 761/1000
Starting Epoch 762/1000
Starting Epoch 763/1000
Starting Epoch 764/1000
Starting Epoch 765/1000
Starting Epoch 766/1000
Starting Epoch 767/1000
Starting Epoch 768/1000
Starting Epoch 769/1000
Starting Epoch 770/1000
Starting Epoch 771/1000
Starting Epoch 772/1000
Starting Epoch 773/1000
Starting Epoch 774/1000
Starting Epoch 775/1000
Starting Epoch 776/1000
Starting Epoch 777/1000
Starting Epoch 778/1000
Starting Epoch 779/1000
Starting Epoch 780/1000
Starting Epoch 781/1000
Starting Epoch 782/1000
Starting Epoch 783/1000
Starting Epoch 784/1000
Starting Epoch 785/1000
Starting Epoch 786/1000
Starting Epoch 787/1000
Starting Epoch 788/1000
Starting Epoch 789/1000
Starting Epoch 790/1000
Starting Epoch 791/1000
Starting Epoch 792/1000
Starting Epoch 793/1000
Starting Epoch 794/1000
Starting Epoch 795/1000
Starting Epoch 796/1000
Starting Epoch 797/1000
Starting Epoch 798/1000
Starting Epoch 799/1000
Starting Epoch 800/1000
Starting Epoch 801/1000
Starting Epoch 802/1000
Starting Epoch 803/1000
Starting Epoch 804/1000
Starting Epoch 805/1000
Starting Epoch 806/1000
Starting Epoch 807/1000
Starting Epoch 808/1000
Starting Epoch 809/1000
Starting Epoch 810/1000
Starting Epoch 811/1000
Starting Epoch 812/1000
Starting Epoch 813/1000
Starting Epoch 814/1000
Starting Epoch 815/1000
Starting Epoch 816/1000
Starting Epoch 817/1000
Starting Epoch 818/1000
Starting Epoch 819/1000
Starting Epoch 820/1000
Starting Epoch 821/1000
Starting Epoch 822/1000
Starting Epoch 823/1000
Starting Epoch 824/1000
Starting Epoch 825/1000
Starting Epoch 826/1000
Starting Epoch 827/1000
Starting Epoch 828/1000
Starting Epoch 829/1000
Starting Epoch 830/1000
Starting Epoch 831/1000
Starting Epoch 832/1000
Starting Epoch 833/1000
Starting Epoch 834/1000
Starting Epoch 835/1000
Starting Epoch 836/1000
Starting Epoch 837/1000
Starting Epoch 838/1000
Starting Epoch 839/1000
Starting Epoch 840/1000
Starting Epoch 841/1000
Starting Epoch 842/1000
Starting Epoch 843/1000
Starting Epoch 844/1000
Starting Epoch 845/1000
Starting Epoch 846/1000
Starting Epoch 847/1000
Starting Epoch 848/1000
Starting Epoch 849/1000
Starting Epoch 850/1000
Starting Epoch 851/1000
Starting Epoch 852/1000
Starting Epoch 853/1000
Starting Epoch 854/1000
Starting Epoch 855/1000
Starting Epoch 856/1000
Starting Epoch 857/1000
Starting Epoch 858/1000
Starting Epoch 859/1000
Starting Epoch 860/1000
Starting Epoch 861/1000
Starting Epoch 862/1000
Starting Epoch 863/1000
Starting Epoch 864/1000
Starting Epoch 865/1000
Starting Epoch 866/1000
Starting Epoch 867/1000
Starting Epoch 868/1000
Starting Epoch 869/1000
Starting Epoch 870/1000
Starting Epoch 871/1000
Starting Epoch 872/1000
Starting Epoch 873/1000
Starting Epoch 874/1000
Starting Epoch 875/1000
Starting Epoch 876/1000
Starting Epoch 877/1000
Starting Epoch 878/1000
Starting Epoch 879/1000
Starting Epoch 880/1000
Starting Epoch 881/1000
Starting Epoch 882/1000
Starting Epoch 883/1000
Starting Epoch 884/1000
Starting Epoch 885/1000
Starting Epoch 886/1000
Starting Epoch 887/1000
Starting Epoch 888/1000
Starting Epoch 889/1000
Starting Epoch 890/1000
Starting Epoch 891/1000
Starting Epoch 892/1000
Starting Epoch 893/1000
Starting Epoch 894/1000
Starting Epoch 895/1000
Starting Epoch 896/1000
Starting Epoch 897/1000
Starting Epoch 898/1000
Starting Epoch 899/1000
Starting Epoch 900/1000
Starting Epoch 901/1000
Starting Epoch 902/1000
Starting Epoch 903/1000
Starting Epoch 904/1000
Starting Epoch 905/1000
Starting Epoch 906/1000
Starting Epoch 907/1000
Starting Epoch 908/1000
Starting Epoch 909/1000
Starting Epoch 910/1000
Starting Epoch 911/1000
Starting Epoch 912/1000
Starting Epoch 913/1000
Starting Epoch 914/1000
Starting Epoch 915/1000
Starting Epoch 916/1000
Starting Epoch 917/1000
Starting Epoch 918/1000
Starting Epoch 919/1000
Starting Epoch 920/1000
Starting Epoch 921/1000
Starting Epoch 922/1000
Starting Epoch 923/1000
Starting Epoch 924/1000
Starting Epoch 925/1000
Starting Epoch 926/1000
Starting Epoch 927/1000
Starting Epoch 928/1000
Starting Epoch 929/1000
Starting Epoch 930/1000
Starting Epoch 931/1000
Starting Epoch 932/1000
Starting Epoch 933/1000
Starting Epoch 934/1000
Starting Epoch 935/1000
Starting Epoch 936/1000
Starting Epoch 937/1000
Starting Epoch 938/1000
Starting Epoch 939/1000
Starting Epoch 940/1000
Starting Epoch 941/1000
Starting Epoch 942/1000
Starting Epoch 943/1000
Starting Epoch 944/1000
Starting Epoch 945/1000
Starting Epoch 946/1000
Starting Epoch 947/1000
Starting Epoch 948/1000
Starting Epoch 949/1000
Starting Epoch 950/1000
Starting Epoch 951/1000
Starting Epoch 952/1000
Starting Epoch 953/1000
Starting Epoch 954/1000
Starting Epoch 955/1000
Starting Epoch 956/1000
Starting Epoch 957/1000
Starting Epoch 958/1000
Starting Epoch 959/1000
Starting Epoch 960/1000
Starting Epoch 961/1000
Starting Epoch 962/1000
Starting Epoch 963/1000
Starting Epoch 964/1000
Starting Epoch 965/1000
Starting Epoch 966/1000
Starting Epoch 967/1000
Starting Epoch 968/1000
Starting Epoch 969/1000
Starting Epoch 970/1000
Starting Epoch 971/1000
Starting Epoch 972/1000
Starting Epoch 973/1000
Starting Epoch 974/1000
Starting Epoch 975/1000
Starting Epoch 976/1000
Starting Epoch 977/1000
Starting Epoch 978/1000
Starting Epoch 979/1000
Starting Epoch 980/1000
Starting Epoch 981/1000
Starting Epoch 982/1000
Starting Epoch 983/1000
Starting Epoch 984/1000
Starting Epoch 985/1000
Starting Epoch 986/1000
Starting Epoch 987/1000
Starting Epoch 988/1000
Starting Epoch 989/1000
Starting Epoch 990/1000
Starting Epoch 991/1000
Starting Epoch 992/1000
Starting Epoch 993/1000
Starting Epoch 994/1000
Starting Epoch 995/1000
Starting Epoch 996/1000
Starting Epoch 997/1000
Starting Epoch 998/1000
Starting Epoch 999/1000
Starting Epoch 1000/1000
```
wandb: - 0.004 MB of 0.004 MB uploaded
wandb: Run history:
wandb:     average_entropy █▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb: average_policy_loss ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█
wandb:      average_reward ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:  average_total_loss ▃█▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:  average_value_loss ▃█▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:               epoch ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:        total_reward ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb: 
wandb: Run summary:
wandb:     average_entropy 0.0
wandb: average_policy_loss 0.0
wandb:      average_reward 0.0
wandb:  average_total_loss 24.44394
wandb:  average_value_loss 24.44394
wandb:               epoch 990
wandb:        total_reward 0
wandb: 
wandb: 🚀 View run Test16-0 at: https://wandb.ai/deeplearning_painn/adversarial_dlr/runs/liuzxvri
wandb: ️⚡ View job at https://wandb.ai/deeplearning_painn/adversarial_dlr/jobs/QXJ0aWZhY3RDb2xsZWN0aW9uOjEyMjczMTczNw==/version_details/v4
wandb: Synced 5 W&B file(s), 0 media file(s), 2 artifact file(s) and 0 other file(s)
wandb: Find logs at: ./wandb/run-20231212_202253-liuzxvri/logs

------------------------------------------------------------
Sender: LSF System <lsfadmin@hpc.dtu.dk>
Subject: Job 19760211: <Test16_0> in cluster <dcc> Done

Job <Test16_0> was submitted from host <gbarlogin2> by user <s230208> in cluster <dcc> at Tue Dec 12 20:22:30 2023
Job was executed on host(s) <4*n-62-20-4>, in queue <gpuv100>, as user <s230208> in cluster <dcc> at Tue Dec 12 20:22:30 2023
</zhome/59/9/198225> was used as the home directory.
</zhome/59/9/198225/Desktop/Adversarial_DRL/Adversarial_DRL> was used as the working directory.
Started at Tue Dec 12 20:22:30 2023
Terminated at Tue Dec 12 20:26:45 2023
Results reported at Tue Dec 12 20:26:45 2023

Your job looked like:

------------------------------------------------------------
# LSBATCH: User input
#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[mem=16G]"
#BSUB -R "select[gpu16gb]"
#BSUB -R "span[hosts=1]"
#BSUB -W 1440
# end of BSUB options
module -s load python3
source ../project-env/bin/activate

python main.py $MYARGS
------------------------------------------------------------

Successfully completed.

Resource usage summary:

    CPU time :                                   187.81 sec.
    Max Memory :                                 821 MB
    Average Memory :                             663.40 MB
    Total Requested Memory :                     65536.00 MB
    Delta Memory :                               64715.00 MB
    Max Swap :                                   -
    Max Processes :                              6
    Max Threads :                                37
    Run time :                                   302 sec.
    Turnaround time :                            255 sec.

The output (if any) is above this job summary.
