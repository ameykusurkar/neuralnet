MNIST_DIR=data/mnist

mnist: $(MNIST_DIR)/training_images $(MNIST_DIR)/training_labels $(MNIST_DIR)/test_images $(MNIST_DIR)/test_labels

mnist_images: $(MNIST_DIR)/training_images
	mkdir -p mnist_images
	generate_mnist_images

$(MNIST_DIR):
	mkdir -p $(MNIST_DIR)

$(MNIST_DIR)/training_images: | $(MNIST_DIR)
	curl http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz > $@.gz
	gzip -d $@.gz

$(MNIST_DIR)/training_labels: | $(MNIST_DIR)
	curl http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz > $@.gz
	gzip -d $@.gz

$(MNIST_DIR)/test_images: | $(MNIST_DIR)
	curl http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz > $@.gz
	gzip -d $@.gz

$(MNIST_DIR)/test_labels: | $(MNIST_DIR)
	curl http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz > $@.gz
	gzip -d $@.gz

clean:
	rm -r $(MNIST_DIR)

.PHONY: mnist clean
