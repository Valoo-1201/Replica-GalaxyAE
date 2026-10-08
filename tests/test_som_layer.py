import numpy as np
from desom_astro.models.som_layer import SOMLayer


def test_forma_y_distancias():
    rng = np.random.default_rng(0)
    z = rng.random((4, 16)).astype('float32')
    capa = SOMLayer(map_size=(5, 5))
    d = capa(z).numpy()

    assert d.shape == (4, 25)
    assert (d >= 0).all()

    protos = capa.get_weights()[0]
    esperado = ((z[:, None, :] - protos[None, :, :]) ** 2).sum(axis=2)
    np.testing.assert_allclose(d, esperado, rtol=1e-4, atol=1e-5)
