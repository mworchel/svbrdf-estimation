import torch

def gamma_decode(images):
    return torch.pow(images, 2.2)

def gamma_encode(images):
    return torch.pow(images, 1.0/2.2)

def pack_svbrdf(normals, diffuse, roughness, specular):
    # We concat on the feature dimension. Here negative in order to handle batches intrinsically-
    return torch.cat([normals, diffuse, roughness, specular], dim=-3)

def unpack_svbrdf(svbrdf, is_encoded = False):
    svbrdf_parts = svbrdf.split(1, dim=-3)

    normals   = None
    diffuse   = None
    roughness = None
    specular  = None
    if not is_encoded:
        normals   = torch.cat(svbrdf_parts[0:3 ], dim=-3)
        diffuse   = torch.cat(svbrdf_parts[3:6 ], dim=-3)
        roughness = torch.cat(svbrdf_parts[6:9 ], dim=-3)
        specular  = torch.cat(svbrdf_parts[9:12], dim=-3)
    else:
        normals   = torch.cat(svbrdf_parts[0:2], dim=-3)
        diffuse   = torch.cat(svbrdf_parts[2:5], dim=-3)
        roughness = torch.cat(svbrdf_parts[5:6], dim=-3)
        specular  = torch.cat(svbrdf_parts[6:9], dim=-3)

    return normals, diffuse, roughness, specular

# We don't really need the encoding...maybe only for testing
# Assumes SVBRDF channels are in range [-1, 1]
def encode_svbrdf(svbrdf):
    normals, diffuse, roughness, specular = unpack_svbrdf(svbrdf, False)

    roughness = roughness.split(1, dim=-3)[0]           # Only retain one channel (roughness if grayscale anyway)
    normals   = torch.cat(normals.split(1, dim=-3)[:2]) # Only retain x and y coordinates of the normal

    return pack_svbrdf(normals, diffuse, roughness, specular)

# Assumes SVBRDF channels are in range [-1, 1]
def decode_svbrdf(svbrdf):
    normals, diffuse, roughness, specular  = unpack_svbrdf(svbrdf, True)

    # Repeat roughness channel three times
    # The weird syntax is due to uniform handling of batches of SVBRDFs and single SVBRDFs
    roughness_repetition     = [1] * len(diffuse.shape)
    roughness_repetition[-3] = 3
    roughness = roughness.repeat(roughness_repetition)

    normals_x, normals_y = normals.split(1, dim=-3)
    normals_x            = normals_x * 3.0
    normals_y            = normals_y * 3.0
    normals_z            = torch.ones_like(normals_x)
    normals              = torch.cat([normals_x, normals_y, normals_z], dim=-3)
    norm                 = torch.sqrt(normals_x**2 + normals_y**2 + normals_z**2)
    normals              = torch.div(normals, norm)

    return pack_svbrdf(normals, diffuse, roughness, specular)

if __name__ == '__main__':
    import math
    import unittest

    class TestGammaFunctions(unittest.TestCase):
        def setUp(self):
            magic_pixel        = 1.3703509847201
            self.encoded_image = [[[magic_pixel]], [[magic_pixel]]]
            self.decoded_image = [[[2.0]], [[2.0]]]

        def test_decode_single(self):
            img = gamma_decode(torch.Tensor(self.encoded_image))
            torch.testing.assert_allclose(img, self.decoded_image)

        def test_decode_batch(self):
            img = gamma_decode(torch.Tensor(self.encoded_image).unsqueeze(0).repeat([5,1,1,1]))
            torch.testing.assert_allclose(img, self.decoded_image)

        def test_encode_single(self):
            img = gamma_encode(torch.Tensor(self.decoded_image))
            torch.testing.assert_allclose(img, self.encoded_image)

        def test_encode_batch(self):
            img = gamma_encode(torch.Tensor(self.decoded_image).unsqueeze(0).repeat([5,1,1,1]))
            torch.testing.assert_allclose(img, self.encoded_image)

    class TestSvbrdfPacking(unittest.TestCase):
        def setUp(self):
            normal_norm    = math.sqrt(3.0)
            normal_xyz     = 1.0 / normal_norm
            self.normals   = torch.Tensor([[[normal_xyz]], [[normal_xyz]], [[normal_xyz]]])
            self.diffuse   = torch.Tensor([[[0.1]], [[0.2]], [[0.3]]])
            self.roughness = torch.Tensor([[[0.3]], [[0.3]], [[0.3]]])
            self.specular  = torch.Tensor([[[0.4]], [[0.5]], [[0.6]]])

            self.batch_size = 5

        def test_pack_single(self):
            svbrdf = pack_svbrdf(self.normals, self.diffuse, self.roughness, self.specular)
            shape  = svbrdf.shape
            self.assertEqual(shape[0], 12) # Channels
            self.assertEqual(shape[1], 1)  # Height
            self.assertEqual(shape[2], 1)  # Width
            torch.testing.assert_allclose(svbrdf[0:3],  self.normals)            
            torch.testing.assert_allclose(svbrdf[3:6],  self.diffuse)
            torch.testing.assert_allclose(svbrdf[6:9],  self.roughness)
            torch.testing.assert_allclose(svbrdf[9:12], self.specular)

        def test_pack_single_encoded(self):
            # TODO: Implement
            self.assertEqual(1, 1)

        def test_pack_batch(self):
            svbrdfs = pack_svbrdf(self.normals, self.diffuse, self.roughness, self.specular).repeat([self.batch_size,1,1,1])
            shape   = svbrdfs.shape
            self.assertEqual(shape[0], self.batch_size) # Batch
            self.assertEqual(shape[1], 12)              # Channels
            self.assertEqual(shape[2], 1)               # Height
            self.assertEqual(shape[3], 1)               # Width
            torch.testing.assert_allclose(svbrdfs[:,0:3],  self.normals)            
            torch.testing.assert_allclose(svbrdfs[:,3:6],  self.diffuse)
            torch.testing.assert_allclose(svbrdfs[:,6:9],  self.roughness)
            torch.testing.assert_allclose(svbrdfs[:,9:12], self.specular)

        def test_pack_batch_encoded(self):
            # TODO: Implement
            self.assertEqual(1, 1)

        def test_unpack_single(self):
            svbrdf = pack_svbrdf(self.normals, self.diffuse, self.roughness, self.specular)
            normals, diffuse, roughness, specular = unpack_svbrdf(svbrdf)
            torch.testing.assert_allclose(normals,   self.normals)
            torch.testing.assert_allclose(diffuse,   self.diffuse)            
            torch.testing.assert_allclose(roughness, self.roughness)
            torch.testing.assert_allclose(specular,  self.specular)

        def test_unpack_single_encoded(self):
            # TODO: Implement
            self.assertEqual(1, 1)

        def test_unpack_batch(self):
            svbrdf = pack_svbrdf(self.normals, self.diffuse, self.roughness, self.specular ).repeat([self.batch_size,1,1,1])
            normals, diffuse, roughness, specular  = unpack_svbrdf(svbrdf)
            self.assertEqual(diffuse.shape[0], self.batch_size)
            self.assertEqual(diffuse.shape[0], self.batch_size)
            self.assertEqual(diffuse.shape[0], self.batch_size)
            self.assertEqual(diffuse.shape[0], self.batch_size)
            torch.testing.assert_allclose(normals,   self.normals)
            torch.testing.assert_allclose(diffuse,   self.diffuse)            
            torch.testing.assert_allclose(roughness, self.roughness)
            torch.testing.assert_allclose(specular,  self.specular)

        def test_unpack_batch_encoded(self):
            # TODO: Implement
            self.assertEqual(1, 1)        

    #class TestSvbrdfPacking(unittest.TestCase):

    unittest.main()