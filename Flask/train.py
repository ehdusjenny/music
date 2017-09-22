import model
import data

def train():
    dg = data.Generator()

    m = model.get_model()
    m.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    data_x = []
    data_y = []
    for _ in range(1000):
        [x,y] = dg.next()
        data_x.push(x)
        data_y.push(y)
    m.fit(data_x,data_y)

    return m
