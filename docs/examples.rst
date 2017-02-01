Examples
=====================================

Here are examples divided in categories.

Basic predictor
----------------

.. code-block:: python

    python examples/commands/basic_predict.py --build-dataset cross --dataset-name basic_test
    python examples/commands/basic_predict.py --train --dataset-name basic_test --model-name svgpc_test --model-version 1
    python examples/commands/basic_predict.py --predict --model-name svgpc_test --model-version 1 --chunk-size 100


Supermarker's numbers tickets transcriptor
---------------------------------------------

The first step is build the number detector from supermarket's tickets (for this is necessary
have installed dlib)

.. code-block:: python

    python examples/commands/hog.py --train
    python examples/commands/hog.py --test
    python examples/commands/hog.py --draw

Then, build the directory with the images for the train dataset

.. code-block:: python

    python examples/commands/tickets2numbers.py --build-images xml

The next step is build the dataset and train the model

.. code-block:: python

    python examples/commands/numbers_clf.py --build-dataset cross --dataset-name numbers_tickets --from-xml
    python examples/commands/numbers_clf.py --train --dataset-name numbers_tickets --model-name numbers_tickets --model-version 1
    python examples/commands/numbers_clf.py --test --model-name numbers_tickets --model-version 1

Finally, run the transcriptor

.. code-block:: python

    python examples/commands/transcriptor.py --model-name numbers_tickets --model-version 1 --transcriptor-ticket-test

Numerai competition
-----------------------

You need to download and save the data in the correspondient folder (previously difined in the settings.cfg file) and run the commands.

.. code-block:: python

    python examples/commands/numerai.py --build-dataset --dataset-name numerai
    python examples/commands/numerai.py --train --dataset-name numerai --ensemble stacking --model-name numerai --model-version 1
    python examples/commands/numerai.py --predict --ensemble stacking --model-name numerai --model-version 1
