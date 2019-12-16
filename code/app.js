(function () {
    new Vue({
        el: "#vueapp",

        data: {
            targetNum: 0,
            trainStatus: "",
            result: ""
        },

        mounted() {
            let c2d = this.drawCanvasContext2d = this.$refs.drawCanvas.getContext("2d");
            c2d.lineWidth = 20;
            c2d.lineCap = "round";
            c2d.lineJoin = "round";

            this.previewCanvasContext2d = this.$refs.previewCanvas.getContext("2d");

            this.loadOrCreateModel();
        },


        methods: {

            async loadOrCreateModel() {
                try {
                    this.model = await tf.loadLayersModel("localstorage://mymodel");
                } catch (e) {
                    console.warn("Can not load model from LocalStorage, so we create a new model");
                    this.model = tf.sequential({
                        layers: [
                            tf.layers.inputLayer({inputShape: [784]}),
                            tf.layers.dense({units: 10}),
                            tf.layers.softmax()
                        ]
                    });
                }

                this.model.compile({
                    optimizer: 'sgd',
                    loss: 'categoricalCrossentropy',
                    metrics: ['accuracy']
                });
            },

            canvasMouseDownHandler(e) {
                this.drawing = true;
                this.drawCanvasContext2d.beginPath();
                this.drawCanvasContext2d.moveTo(e.offsetX, e.offsetY);
            },

            canvasMouseMoveHandler(e) {
                if (this.drawing) {
                    this.drawCanvasContext2d.lineTo(e.offsetX, e.offsetY);
                    this.drawCanvasContext2d.stroke();
                }
            },

            canvasMouseUpHandler(e) {
                this.drawing = false;

                this.previewCanvasContext2d.fillStyle = "white";
                this.previewCanvasContext2d.fillRect(0, 0, 28, 28);
                this.previewCanvasContext2d.drawImage(this.$refs.drawCanvas, 0, 0, 28, 28);
            },

            btnClearCanvasClickedHandler(e) {
                this.drawCanvasContext2d.clearRect(0, 0, this.$refs.drawCanvas.width, this.$refs.drawCanvas.height);
            },

            getImageData() {
                let imageData = this.previewCanvasContext2d.getImageData(0, 0, 28, 28);
                let pixelData = [];

                let color;
                for (let i = 0; i < imageData.data.length; i += 4) {
                    color = (imageData.data[i] + imageData.data[i + 1] + imageData.data[i + 2]) / 3;
                    pixelData.push(Math.round((255 - color) / 255));
                }
                return pixelData;
            },


            async btnTrainClickedHandler(e) {
                let data = this.getImageData();
                console.log(this.targetNum);
                // [1,0,0,0,0,0,0,0,0,0]
                // [0,1,0,0,0,0,0,0,0,0]
                let targetTensor = tf.oneHot(parseInt(this.targetNum), 10);

                let self = this;
                console.log("Start training");
                await this.model.fit(tf.tensor([data]), tf.tensor([targetTensor.arraySync()]), {
                    epochs: 30,
                    callbacks: {
                        onEpochEnd(epoch, logs) {
                            console.log(epoch, logs);
                            self.trainStatus = `<div>Step: ${epoch}</div><div>Loss: ${logs.loss}</div>`;
                        }
                    }
                });
                self.trainStatus = `<div style="color: green;">训练完成</div>`;
                console.log("Completed");

                await this.model.save("localstorage://mymodel");
            },

            async btnPredictClickedHandler(e) {
                let data = this.getImageData();
                let predictions = await this.model.predict(tf.tensor([data]));
                this.result = predictions.argMax(1).arraySync()[0];
            }
        }
    });
})();