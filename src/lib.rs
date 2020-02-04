//! An implementation of the quasirandom _Rd_ sequence described [in this blog
//! post][1] via a fast and accurate fixed-point representation correct for up
//! to `2^64` samples; samples can be produced with one 128-bit multiplication
//! per dimension and are guaranteed accurate to within one part per `2^64`.
//!
//! [1]: http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/

#![forbid(missing_docs, unsafe_code)]

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, ToPrimitive};

/// A d-dimensional quasirandom Rd sequence.
///
/// Successive samples are generated dimension by dimension, so every `d`
/// outputs generated constitute one individual sample from the sequence.
#[derive(Clone, Debug)]
pub struct Sequence {
    parameters: Box<[u128]>,
    dimensionality: usize,
    sample: u64,
    dimension: usize,
}

impl Sequence {
    /// Creates a new quasirandom Rd sequence.
    pub fn new(dimensionality: usize) -> Self {
        Self::new_with_offset(dimensionality, 0)
    }

    /// Creates a new sequence and specifies the starting sample offset.
    pub fn new_with_offset(dimensionality: usize, sample: u64) -> Self {
        let mut parameters = vec![0; dimensionality].into_boxed_slice();

        generate_parameters(dimensionality, &mut parameters);

        Self {
            parameters,
            dimensionality,
            sample,
            dimension: 0,
        }
    }

    /// Seeks to the given sample offset. Starts at dimension 0 of the given sample.
    pub fn seek_to(&mut self, sample: u64) {
        self.sample = sample;
        self.dimension = 0;
    }
    
    /// Seeks a relative offset of samples. Starts at dimension 0 of the sample which is seeked to.
    /// Performs a wrapping addition.
    pub fn seek(&mut self, offset: u64) {
        self.sample.wrapping_add(offset);
        self.dimension = 0;
    }
    
    /// Seeks to the given dimension within the current sample.
    pub fn seek_to_dimension(&mut self, dimension: usize) {
        assert!(dimension < self.dimensionality);
        self.dimension = dimension;
    }
    
    /// Seeks a relative offset of dimensions within the current sample.
    pub fn seek_dimension(&mut self, offset: usize) {
        let dimension = self.dimension + offset;
        assert!(dimension < self.dimensionality);
        self.dimension = dimension;
    }

    /// Generates the next sample dimension as a [0, 1) fixed-point value.
    pub fn next_u64(&mut self) -> u64 {
        raw_to_u64(self.next_raw())
    }

    /// Generates the next sample dimension as a [0, 1) fixed-point value.
    pub fn next_u32(&mut self) -> u32 {
        raw_to_u32(self.next_raw())
    }

    /// Generates the next sample dimension as a [0, 1) floating-point value.
    pub fn next_f64(&mut self) -> f64 {
        raw_to_f64(self.next_raw())
    }

    /// Generates the next sample dimension as a [0, 1) floating-point value.
    pub fn next_f32(&mut self) -> f32 {
        raw_to_f32(self.next_raw())
    }
    
    /// Fills the slice with the next sample dimensions in the sequence. Will wrap to dimensions of the next sample
    /// after the current sample has been exhausted.
    pub fn fill_with_samples_f32(&mut self, samples: &mut [f32]) {
        for v in samples.iter_mut() {
            *v = self.next_f32();
        }
    }
    
    /// Fills the slice with the next sample dimensions in the sequence. Will wrap to dimensions of the next sample
    /// after the current sample has been exhausted.
    pub fn fill_with_samples_f64(&mut self, samples: &mut [f64]) {
        for v in samples.iter_mut() {
            *v = self.next_f64();
        }
    }
    
    /// Fills the slice with the next sample dimensions in the sequence. Will wrap to dimensions of the next sample
    /// after the current sample has been exhausted.
    pub fn fill_with_samples_u32(&mut self, samples: &mut [u32]) {
        for v in samples.iter_mut() {
            *v = self.next_u32();
        }
    }
    
    /// Fills the slice with the next sample dimensions in the sequence. Will wrap to dimensions of the next sample
    /// after the current sample has been exhausted.
    pub fn fill_with_samples_u64(&mut self, samples: &mut [u64]) {
        for v in samples.iter_mut() {
            *v = self.next_u64();
        }
    }
    
    /// Get the raw bit pattern of the next dimension in the sequence. Note that this value
    /// shouldn't be used directly as a numerical value. Use `raw_to_[u32/u64/f32/f64]` functions
    /// to convert this value to a number.
    ///
    /// Getting the raw value is mostly useful for 'scrambling', where you would XOR the raw bit
    /// pattern with a scramble value before converting it to a useable number.
    pub fn next_raw(&mut self) -> u64 {
        let output = self.parameters[self.dimension].wrapping_mul(self.sample as u128);

        if self.dimension + 1 == self.dimensionality {
            self.sample = self.sample.wrapping_add(1);
            self.dimension = 0; // set to next sample
        } else {
            self.dimension += 1;
        }

        (output >> 64) as u64
    }


}

fn shift_round(value: u64, n: u32) -> u64 {
    (value + (1 << (n - 1))) >> n
}

/// Convert a raw sample bit pattern to an f32 in [0, 1).
pub fn raw_to_f32(raw: u64) -> f32 {
    (0.5 + f32::from_bits(0x3F80_0000 | shift_round(raw, 41) as u32)).fract()
}

/// Convert a raw sample bit pattern to an f64 in [0, 1).
pub fn raw_to_f64(raw: u64) -> f64 {
    (0.5 + f64::from_bits(0x3FF0_0000_0000_0000 | shift_round(raw, 12))).fract()
}

/// Convert a raw sample bit pattern to a 32-bit fixed-point number in [0, 1).
pub fn raw_to_u32(raw: u64) -> u32 {
    (1u32 << 31).wrapping_add(shift_round(raw, 32) as u32)
}

/// Convert a raw sample bit pattern to a 64-bit fixed-point number in [0, 1).
pub fn raw_to_u64(raw: u64) -> u64 {
    (1u64 << 63).wrapping_add(raw)
}

/// Scramble a raw sequence bit pattern with a scramble value and get the resulting f32 number in [0, 1).
pub fn scramble_f32(raw: u64, scramble: u64) -> f32 {
    raw_to_f32(raw ^ scramble)
}

/// Scramble a raw sequence bit pattern with a scramble value and get the resulting f32 number in [0, 1).
pub fn scramble_f64(raw: u64, scramble: u64) -> f64 {
    raw_to_f64(raw ^ scramble)
}

/// Scramble a raw sequence bit pattern with a scramble value and get the resulting
/// 32-bit fixed-point number in [0, 1).
pub fn scramble_u32(raw: u64, scramble: u64) -> u32 {
    raw_to_u32(raw ^ scramble)
}

/// Scramble a raw sequence bit pattern with a scramble value and get the resulting
/// 64-bit fixed-point number in [0, 1).
pub fn scramble_u64(raw: u64, scramble: u64) -> u64 {
    raw_to_u64(raw ^ scramble)
}

/// Populates a buffer with all parameters for a given dimensionality.
///
/// If you intend to generate sequence samples yourself, the `i`th parameter in
/// the buffer will equal `2^128 / phi^(i + 1)` rounded to the nearest integer.
pub fn generate_parameters(dimensionality: usize, buffer: &mut [u128]) {
    assert!(dimensionality > 0 && buffer.len() == dimensionality);
    assert!(dimensionality <= 1_000_000, "max dimensionality 1M");

    let (phi, compute_precision) = compute_phi(dimensionality);
    let scale = BigInt::one() << 128; // fixed-point parameters

    // Generate the parameters for all dimensions as successive powers of the
    // reciprocal of `phi_d` using the appropriate precision for correctness.

    let mut power = phi.clone();

    for parameter in buffer.iter_mut() {
        *parameter = (power.recip() * &scale).round().numer().to_u128().unwrap();
        power = (power * &phi * &compute_precision).round() / &compute_precision;
    }
}

fn fixed_point_pow(mut x: BigRational, scale: &BigInt, mut n: usize) -> BigRational {
    let mut result = BigRational::one();

    while n > 0 {
        if n & 1 != 0 {
            result = ((result * &x) * scale).round() / scale;
        }

        x = ((&x * &x) * scale).round() / scale;

        n >>= 1;
    }

    result
}

fn compute_phi(dimensionality: usize) -> (BigRational, BigInt) {
    let mut xi = BigRational::one();

    // TODO: required precision for the Newton-Raphson stage has not been fully
    // analyzed yet; use 200-bit precision just to be sure, but it is too much.

    let one = BigRational::one();
    let precision = one.numer() << 200;
    let n = BigInt::from(1 + dimensionality);

    for _ in 0..12 {
        let xi_n_minus_1 = fixed_point_pow(xi.clone(), &precision, dimensionality);

        let derv = &xi_n_minus_1 * &n - &one;
        let func = (xi_n_minus_1 - &one) * &xi - &one;

        // Generate the next Newton-Raphson approximation then immediately round it
        // to a multiple of `precision` to keep intermediate results under control.

        xi = ((&xi - func / derv) * &precision).round() / &precision;
    }

    (xi, (one.numer() << 132) * (dimensionality + 1))
}

#[cfg(test)]
mod tests {
    use super::Sequence;

    #[test]
    fn test_2d_u64() {
        let mut seq = Sequence::new(2);

        assert_eq!(seq.next_u64(), 9223372036854775808);
        assert_eq!(seq.next_u64(), 9223372036854775808);
        assert_eq!(seq.next_u64(), 4701663079357100687);
        assert_eq!(seq.next_u64(), 1288325974074489628);
    }

    #[test]
    fn test_2d_f64() {
        let mut seq = Sequence::new(2);

        assert_eq!(seq.next_f64(), 0.5);
        assert_eq!(seq.next_f64(), 0.5);
        assert_eq!(seq.next_f64(), 0.25487766624669295);
        assert_eq!(seq.next_f64(), 0.06984029099805333);
    }

    #[test]
    fn test_offset() {
        let mut seq1 = Sequence::new(1);
        let mut seq2 = Sequence::new_with_offset(1, 100);

        for _ in 0..100 {
            seq1.next_u64();
        }

        assert_eq!(seq1.next_u64(), seq2.next_u64());
        assert_eq!(seq1.next_u64(), seq2.next_u64());
    }

    // Reference tests against a SageMath computation of parameters for various
    // dimensionalities carried out at 1024-bit precision with the mpf package.

    #[test]
    fn check_reference_parameters_1d() {
        let seq = Sequence::new(1);

        assert_eq!(seq.parameters[0], 0x9e3779b97f4a7c15f39cc0605cedc834);
    }

    #[test]
    fn check_reference_parameters_2d() {
        let seq = Sequence::new(2);

        assert_eq!(seq.parameters[0], 0xc13fa9a902a6328f434ff71b2d97724b);
        assert_eq!(seq.parameters[1], 0x91e10da5c79e7b1cd438a0a8e6c9c0fc);
    }

    #[test]
    fn check_reference_parameters_3d() {
        let seq = Sequence::new(3);

        assert_eq!(seq.parameters[0], 0xd1b54a32d192ed039fdcaa7d6e9dfe0f);
        assert_eq!(seq.parameters[1], 0xabc98388fb8fac028c2245930757dd50);
        assert_eq!(seq.parameters[2], 0x8cb92ba72f3d8dd732e5bf2dbe798195);
    }

    #[test]
    fn check_reference_parameters_5d() {
        let seq = Sequence::new(5);

        assert_eq!(seq.parameters[0], 0xe19b01aa9d42c6333587690489ecb903);
        assert_eq!(seq.parameters[1], 0xc6d1d6c8ed0c9631c8f17386e81d96a8);
        assert_eq!(seq.parameters[2], 0xaf36d01ef7518dbbd36071fc91bf9f6e);
        assert_eq!(seq.parameters[3], 0x9a69443f36f710e68a65d16792971bec);
        assert_eq!(seq.parameters[4], 0x881403b9339bd42dfd680e7ae5211382);
    }

    #[test]
    fn check_reference_parameters_50d() {
        #[rustfmt::skip]
        const REFERENCE: &[u128] = &[
            0xfc8296a209364b84342308d1429bd922, 0xf9115b2c5756892c42754a2f380e0ce3, 0xf5ac231dc3dba58bb2d705886af43014, 0xf252c4897e427e1bfa398ad678c89f07, 
            0xef051615065ccebe1e8a792351b5c0e2, 0xebc2eef62db2c107dce7f0bb56c69963, 0xe88c26f11fda7c6a57bd604ac36c610e, 0xe560965671ad6c33e18c6ddd2acce765, 
            0xe240160137434536124e925a36d1cdca, 0xdf2a7f55209b28790f5ed35fa68778a3, 0xdc1fac3c9cdb92e0d146cbcc4b4c0acd, 0xd91f772704121b053a1a920a62c4b881, 
            0xd629bb06c75c4ebde6dd4aaab058fb4a, 0xd33e534fa763510aad9bac8821f42a43, 0xd05d1bf4f114271581e79affc3cec797, 0xcd85f167c07ef00dc63a740d68845005, 
            0xcab8b09549c78f9399687452a6fe60c3, 0xc7f536e528129d571db5b0aca48c07ad, 0xc53b6237b259b677a87245a274a9da8d, 0xc28b10e456129614cdd8652a970a4d17, 
            0xbfe421b7f794a26ad95bfe8ba115147f, 0xbd4673f35828e2c0ee8ba75c57e88792, 0xbab1e74981b0986435cd7920ec073d59, 0xb8265bde37cef9ec7a78c7c15b3aae16, 
            0xb5a3b2446e82d416d11850ffea6aa29e, 0xb329cb7cc61d16a5856baf9954985cf4, 0xb0b888f40c8195eced015efcd3cedf28, 0xae4fcc81c39f8b05fea6613dc848d5c8, 
            0xabef7866ad0f9d0cdef1fc6e46600114, 0xa9976f4b5ac57d5e0a68bbb9581fa0d6, 0xa747943ec4c35f7376bf574006e06096, 0xa4ffcab4e3bdd2d6f4a757ef78ddc740, 
            0xa2bff685509ec29b45b21b31a7af10ae, 0xa087fbe9e8d699f4af40ddec8f5ed96b, 0x9e57bf7d776ac8df3aeb50b7e6ceb82e, 0x9c2f263a62b11e472ae3675a1a187f18, 
            0x9a0e15795ea7a6e240494d6a74f9ceb5, 0x97f472f023d8f8d912e86968c659e183, 0x95e224b02abd0c89a4894e1253573542, 0x93d711256b86fa124361ef4b8c7be216, 
            0x91d31f15225029f8633eb85dcd76ab01, 0x8fd6359c9791bd24f67a5a349e408c8b, 0x8de03c2fecdd25999842c468b8fb3ed7, 0x8bf11a98edc51db219709c807eede0ea, 
            0x8a08b8f5e4e85f7f3fd9a52c8b3f2739, 0x8826ffb8750fb0cc44302fcbb9039b9b, 0x864bd7a476510ab34a2752a191b2decf, 0x847729ced729d5481f1426f2513c3ac9, 
            0x82a8df9c818260d6618630c73dfe6938, 0x80e0e2c1438cf67b2cdfbf859f814657, 
        ];

        for (index, &parameter) in Sequence::new(50).parameters.iter().enumerate() {
            assert_eq!(parameter, REFERENCE[index]);
        }
    }

    #[test]
    fn check_reference_parameters_272d() {
        #[rustfmt::skip]
        const REFERENCE: &[u128] = &[
            0xff5982a853359d81dce8b2829f9f1426,0xfeb37197717e0e3fffc59b976d75c340,0xfe0dcc86efedfdb91101d10b247fb334,0xfd6893309165e6574aa5a50ac9094741,
            0xfcc3c54e467448dc1860be901dd1f44f,0xfc1f629a2d37f72a5cee22d95c81e981,0xfb7b6ace91427262c3f71e9fe0a74cc1,0xfad7dda5eb7a5c458fcb46c2167978dc,
            0xfa34badae1fdfbcd5a6fbd1d6b551052,0xf99202284805d4f649ae8adc44083f61,0xf8efb3491dc753a53df038f6ea7b340c,0xf84dcdf8905789a28bc5dd878776f06c,
            0xf7ac51f1f98dff9bd8209c48e6bcfbfa,0xf70b3ef0dfe79920b7451d940742b95e,0xf66a94b0f6698b8db695c4dd301a0875,0xf5ca52ee1c8467d9815579780ba9a4ff,
            0xf52a79645df73737d886a5b28d749f87,0xf48b07cff2b2aa861e06ac85533c13db,0xf3ebfded3ebc5c753afd8090c3e49d00,0xf34d5b78d2122664b1ab43412c866081,
            0xf2af202f688d87e2b28ae4201b4ea40f,0xf2114bcde9c720c514a8a1ea9c4598d0,0xf173de1168fa3dcf18ee1454f9fd32bc,0xf0d6d6b724e877d7e80309cd41df2f6e,
            0xf03a357c87bd6565c23b0b63e2c264f5,0xef9dfa1f26f25eb1e0dac598fad67ad7,0xef02245cc33254090fceea76d421c4d7,0xee66b3f3483db67d0eb7636e0093d204,
            0xedcba8a0ccce72dacfeddb22b4d0af28,0xed310223927bfed9b3edcc107a671d7e,0xec96c03a059f7877e73e5be201e8cb60,0xebfce2a2bd37d77810b35fdd8b3cb1cd,
            0xeb63691c7ace30f4858af817ace685a6,0xeaca53662a5a0cfc40984080c02a9444,0xea31a13ee225ce2de151ab954777a1c4,0xe9995265e2b32b44ff3ab5bd9089f8da,
            0xe901669a969fba8e25bcc578ca82b2bc,0xe869dd9c92898f35d41a3e85dd4fbdb4,0xe7d2b72b94f3e866e4b910907d9a91a5,0xe73bf307862bf22cc68f5fb121834992,
            0xe6a590f0782d980dfb05517e6aaf2faa,0xe60f90a6a688695352228fb8087835b5,0xe579f1ea76448ef0665cb4e5a45cf1a5,0xe4e4b47c75c7d301e0d59acdabc21143,
            0xe44fd81d5cbab9d6164d69933c2045c9,0xe3bb5c8e0bedac74947c53c18993e2cd,0xe327418f8d3e34993ef322b52cba8ab4,0xe29386e3137c4a17a2081ad6301797b4,
            0xe2002c49fa4fb19b29b75313bf52e15d,0xe16d3185c61d6cb8f1ba622d419e26f1,0xe0da965823ed3b47eb7341aba15c54c2,0xe0485a82e94f2df41d997d1dbafff667,
            0xdfb67dc814414a01c8e7374f8abeb729,0xdf24ffe9cb153e35444d41e5d5d1a7d9,0xde93e0aa5c5628d46a7a791ed657cab5,0xde031fcc3eae6eb679c2d088fbe42146,
            0xdd72bd1210cda3574eae015e386a0f85,0xdce2b83e994e81e2e8ac9aedabd5f43d,0xdc531114c69cf72e2fa45425ebd086b0,0xdbc3c757aedc3c91f831edf4bf4ab3f2,
            0xdb34daca8fcd039d4ba7bbd910d60faf,0xdaa64b30ceb3b293fff405ba3f1e690d,0xda18184df83eb1aeb2b9eee3fe4a746f,0xd98a41e5c06cc911420264ece5aef925,
            0xd8fcc7bc02738f6ce400d152ca512744,0xd86fa994c0a5e9430678e4bc71d4a06d,0xd7e2e734245a98be255fd6134ff4ea25,0xd756805e7dd2de15cf5bdd0744dbfc48,
            0xd6ca74d84421287315c88a05250f792e,0xd63ec466150fd74aace5e830b4522320,0xd5b36eccb5080c2207d313833c7fc34e,0xd52873d10ef88cb4c2eb20beaae39967,
            0xd49dd338343cb56fb60ce165d7e6b355,0xd4138cc75c837c370f43364f18cc4abe,0xd389a043e5b6836cdd2c4a9e59e41a58,0xd3000d7353e13d2d776239edb887882f,
            0xd276d41b51181eb73a075212f702e94e,0xd1edf401ad5fe3f31073562f5a5c02f3,0xd1656cec5e94e31351d5e876ce9a3f57,0xd0dd3ea18052703e79748d40f1a35e2c,
            0xd05568e753da513b4af897675bf2a8fb,0xcfcdeb843ffc4113fa0abe92c4df2ba6,0xcf46c63ed0fd83a5f33f26c530eafa7d,0xcebff8ddb8808913eb1540090fabf664,
            0xce398327cd6ca10fde8b20910a5520bb,0xcdb364e40bd5bdf2b77bd59d651a26c8,0xcd2d9dd994e447974db3a30382737755,0xcca82dcfaebcfdee84544911bb45b56f,
            0xcc23148dc468eb4249ce2f92ec9d86a8,0xcb9e51db65bd661d4759c5bd8db224af,0xcb19e580474422cd137e84b3bb29f7c8,0xca95cf4442235474c1d2cfb475a5c1df,
            0xca120eef5405dda5b0b96d08938f8a91,0xc98ea4499f0390747c7687dc00282940,0xc90b8f1b69897e000587186068fcfdf1,0xc888cf2d1e4255607eb2379233c08cd5,
            0xc80664474bfed1f47dd4498a75d94d8e,0xc7844e32a59e390210e80f35d3603c13,0xc7028cb801f6e6a1df519953fe10890c,0xc6811fa05bbee9ea65eac58f9320dcef,
            0xc60006b4d174b05162c8660a13979a6f,0xc57f41bea547c0398c2276a19e285ec2,0xc4fed0873d0182a2b438d43a1c449db5,0xc47eb2d821ee1bf28278cc2c26a07e89,
            0xc3fee87b00c553cbf08e8f61a2a6f53d,0xc37f7139a9938bebc0702051dfccd245,0xc3004cde0fa2c60026cdb7b1e312998e,0xc2817b324963b871ebaddbfcef3ecd13,
            0xc202fc009056f215495186a6c6bf878a,0xc184cf1340f60cb8d7ceb184758be352,0xc106f434da9cee88da1d8a6fbb5008b7,0xc0896b2fff731a3d479e5508c7fc4ceb,
            0xc00c33cf74550e08f363b56c80bb48bb,0xbf8f4dde20bdb14038cebd63cfb0bca0,0xbf12b9270eafd0ad9b47a2a1b93a6ff3,0xbe9675756a9fa98acd1782bf7f9665ab,
            0xbe1a8294835c8314979d0b6bb319dc8f,0xbd9ee04fc9fa56b025474554a365eaa9,0xbd238e72d1bb869833ef215880ca68cb,0xbca88cc94ffaa308bc51c4142aa2ae4b,
            0xbc2ddb1f1c143ddfa192e6a202250d83,0xbbb379402f50cca901cffed7481cd7f1,0xbb3966f8a4ce990dc6ea402a04b39423,0xbabfa414b96bbf9b1cc6eb3aff5dbb25,
            0xba463060cbb03cd9785acf5cf7180b0c,0xb9cd0ba95bb808a9e0e759f07eb4cc5e,0xb95435bb0b1d3fe032dd159d56d72c67,0xb8dbae629ce25c1119e111e7e3d283b5,
            0xb863756cf55c798a857854ead210c0dd,0xb7eb8aa71a1dab6d61de37a7d6c91b37,0xb773edde31df5ddf648863faca6cdd8a,0xb6fc9edf846cc64bc1d51c4a83f18f90,
            0xb6859d787a8d61a9a75683239eb7b0c6,0xb60ee9769def80bf5c1fc681c903f511,0xb59882a79912e258ee676231d2c7fcef,0xb52268d937335b695bbc20ed43fb19cf,
            0xb4ac9bd964338d0d26f11f19f47be0be,0xb4371b762c87a86554c7d6f581343eda,0xb3c1e77dbd204041ce3e2ee476434db6,0xb34cffbe635528922d41a21ce7fcc4ac,
            0xb2d864068cd06393fe5fec6fa9e0eeaf,0xb2641424c7791cb587e126046b2342db,0xb1f00fe7c15eb1242c86f5ec2f66276b,0xb17c571e48a3c5fd85f874f1b82c7d3a,
            0xb108e9974b696c1a58aa877dbe0cc362,0xb095c721d7ba516a88d8e030f49b223a,0xb022ef8d1b75ffd93de48771cdc4972c,0xafb062a8643c29b06729b1083e4c4ca8,
            0xaf3e20431f580372db17da6d447d8a54,0xaecc282cd9abab244f0ca51bee1389f1,0xae5a7a353f9b9cf66c23a261c8e989a2,0xade9162c1cfa35524adb544f2e45da01,
            0xad77fbe15cf34035a51a03e5438e335e,0xad072b2509f795db13c3b9dedcfa2e9d,0xac96a3c74da8c4a4b2b5a7d8178308f1,0xac26659870c4c8408b9aa58a600389d7,
            0xabb67068db11cdfe2da813b2096e0eab,0xab46c409134a064cddea7e60517ab523,0xaad76049bf078359d15dbd57fc9306bd,0xaa6844fba2b024c5e79d14f93680e344,
            0xa9f971efa161906a6284ff8f4b10e3a7,0xa98ae6f6bcdd38241ca9cfe2ad120db5,0xa91ca3e215746c9cc60c4d1cb7a1da50,0xa8aea882e9f47d09b3f7bfa82b56e84b,
            0xa840f4aa9792e3d7d673a91fa6856040,0xa7d3882a99d9803c6b2e90dec07f3d20,0xa76662d48a92dca20b3eedbe026096ed,0xa6f9847a21b681e9b68b463730c0359c,
            0xa68ceced35555777851b24f63299936b,0xa6209bffb9861002abff7b01d41385cc,0xa5b49183c051a32088ea7c7ba87ed3e3,0xa548cd4b799fd3826cf4f3c2e018fd81,
            0xa4dd4f293323c1dde47172d4ffc7806a,0xa47216ef58488c773f0cb6618a04d6a7,0xa4072470721dfb4620d5fb71d35f938d,0xa39c777f274538abea21fa09f9d18a13,
            0xa3320fee3bdd96b3c98faf07524238b2,0xa2c7ed90917160d450c71ffe2ee9938a,0xa25e103926e2ba2a69d7cf35bfe40ef6,0xa1f477bb185888259066ae5ca1b13cd9,
            0xa18b23e99f2b699d37220669a1bf9085,0xa122149811d2ba474735048f105b0406,0xa0b94999e3d1a287acb4696ee5a23c32,0xa050c2c2a5a43390e83a3b14dd593e38,
            0x9fe87fe604ac8fcda31f55d9d9c045a2,0x9f8080d7cb201f8b48f74e704493e22d,0x9f18c56bdff4d1ddae254a2942e95334,0x9eb14d7646ce69b3d08f443a70d40191,
            0x9e4a18cb1febd715c4a0a99e19efe321,0x9de3273ea8149c83e5f54b45b373e82b,0x9d7c78a53886406f682a61ef7dbb41dd,0x9d160cd346e1cac46974c232fca62326,
            0x9cafe39d65194e7dadb96c71530efaca,0x9c49fcd8415d7f3a2e005d71feeb398a,0x9be45858a60b52ccad30ebbc1153fee2,0x9b7ef5f37999aebd881b155877164412,
            0x9b19d57dbe8721b6fbe1ebc2838db311,0x9ab4f6cc9347a8d422e8ca8691ed0b98,0x9a5059b5323280caed6f3a8fb49120b7,0x99ebfe0cf17002e9600e529025b98b10,
            0x9987e3a942e78dde674e0e663460baf5,0x99240a5fb42d7a4694897e010b6337e6,0x98c07205ee711af51d55d234df2bf7ae,0x985d1a71b66ac8f17b9a35425411bf4a,
            0x97fa0378ec49fb22117d07ad7aa48fe5,0x97972cf18ba3699c393e8a785d3990e7,0x973496b1ab5f3c922e0938053a85beb1,0x96d2408f7da746d73fad0ee53dfc8276,
            0x96702a614fd54bf2c824dfc0853dbf18,0x960e53fd8a6151ba5eab4b68e3b5f767,0x95acbd3ab0cffd6bc9088a1e195cea53,0x954b65ef61a0fc3f30a25322bbf1a91c,
            0x94ea4df2563d776a25b44e0e2eca71ee,0x9489751a62e6938bfff15fd9a7ac0131,0x9428db3e76a3fb7c30a4e68b0326c740,0x93c880359b3276731f3e91a5563cae3a,
            0x936863d6f4f289862e03fd38ad7fc3bc,0x930885f9c2d7246f89708b92dea2f65a,0x92a8e6755e54599a6996303bffeb6e33,0x924985213b4e216c7198f819f560a877,
            0x91ea61d4e80728c4de231856302a36fa,0x918b7c680d0faaaa387031f48a1791d8,0x912cd4b26d34551f484c44f61644cd06,0x90ce6a8be56d3918041d8569810556b0,
            0x90703dcc6cccc58742c7e8ed8ed09158,0x90124e4c146ecd7df7edd5e9881696fa,0x8fb49be307679954c7c2e119add951e9,0x8f5726698ab302d8c453fc00ababb43d,
            0x8ef9edb7fd239c742ad3e1568efb4e8c,0x8e9cf1a6d751e34bfc22eda4fe753e0b,0x8e40320eab8b7c4b506eebe0da72f907,0x8de3aec825c27c154a69b1f36b581a59,
            0x8d8767ac0b7cb9d69334b9bfd75d95fc,0x8d2b5c933bc32cef4cbc323a89ff2276,0x8ccf8d56af11556e6cd55394015908c3,0x8c73f9cf7744af57780a133a50fca2cc,
            0x8c18a1d6bf8c30ab978faa7e892bafdb,0x8bbd8545cc57d22f0a75beedc441fc72,0x8b62a3f5fb4822e3f6a95fd84e9082b8,0x8b07fdc0c31de633a2f17b02591142a4,
            0x8aad927fb3a9bcbf2791e4fdd5ca88f8,0x8a53620c75bbd7d0a7c69a202585ef8a,0x89f96c40cb13b7662aca755289fe16e7,0x899fb0f68e4ff2cf2f9b440b073935a9,
            0x89463007b2de0bd61b28dc746db10a40,0x88ece94e44ea4c6ea515ae165b1bfb42,0x8893dca4694faee26ba5300e8a95a760,0x883b09e45d87d074dbe79408043ffb67,
            0x87e270e8779aee779fa244759b125a88,0x878a118b260fedc8c6e1f10d6e209e78,0x8731eba6efdc6cb3e78d47fb5896fe3a,0x86d9ff167454df2f71b6f697afde7ca6,
            0x86824bb46b1cb56f7bd22b847903be49,0x862ad15ba4168cc74e4e97c23dcf5888,0x85d38fe707546ad2fa7fd672891f334b,0x857c8731950802e24d10357345bfa1b3,
            0x8525b7166573059e7098159d03738f19,0x84cf1f70a8d77ae39949820e8f79567d,0x8478c01ba76825c816f33469321238f2,0x842298f2c138f2ca2eef0207e89a09cd,
            0x83cca9d16e2f701e23dda5e591584d32,0x8376f2933df35015d55d05e5787eb22e,0x83217313d7def59b662e725921f9f065,0x82cc2b2efaf00ab85b87f5be6feb17a0,
            0x82771ac07db82122ac8e9a9a1990eef1,0x822241a44e4d5cc93d3797d2544787d1,0x81cd9fb6723b2859440b9ad1f56c06af,0x817934d30672f3b61e81ebd161b3c46c,
            0x812500d63f3cfc5d1be0edb35ae6f88c,0x80d1039c68291faecac681ba03700df8,0x807d3d01e3ffb71759ad21bcf18d17d3,0x8029ace32cb27e0f9ef2134372e1305a,
        ];

        for (index, &parameter) in Sequence::new(272).parameters.iter().enumerate() {
            assert_eq!(parameter, REFERENCE[index]);
        }
    }

    #[test]
    #[ignore]
    fn check_reference_parameters_40000d() {
        let seq = Sequence::new(40000);

        assert_eq!(seq.parameters[0], 0xfffedd47769e0240104c06f7343b7241);
        assert_eq!(seq.parameters[9], 0xfff4a504aa45d10d26450725c843317c);
        assert_eq!(seq.parameters[99], 0xff8e88d67f9a0aafcb869a6de63d35eb);
        assert_eq!(seq.parameters[18930], 0xb867948c981e001579b7c46882c694b9);
        assert_eq!(seq.parameters[37919], 0x84b2e6946df6ad7c7c968dae86083dd4);
        assert_eq!(seq.parameters[39998], 0x8000da0b87eca68f3ee760b18f991c7d);
        assert_eq!(seq.parameters[39999], 0x800048ae4b9d6a67de0a620ebab103c6);
    }

    #[test]
    #[ignore]
    fn check_reference_parameters_1000000d() {
        let seq = Sequence::new(1000000);

        assert_eq!(seq.parameters[0], 0xfffff45ef542aa0acd04b35a993aa9a8);
        assert_eq!(seq.parameters[10], 0xffff8014a6eb3b2225d711ecd213d01c);
        assert_eq!(seq.parameters[100], 0xfffb69812ef796cc3756e4b28aeced9f);
        assert_eq!(seq.parameters[1000], 0xffd28b55902c3b44798d606fa56be3fa);
        assert_eq!(seq.parameters[10000], 0xfe3b43d01c0595637b7b2c71a4aa6378);
        assert_eq!(seq.parameters[100000], 0xeedb35b9f1f390caa3abbde7ee8ec634);
        assert_eq!(seq.parameters[162834], 0xe4ad372d72d7b768c9241bf29896218f);
        assert_eq!(seq.parameters[528192], 0xb18428c86a7ad3d9a5f8a74daad3ea02);
        assert_eq!(seq.parameters[819383], 0x9112341a999df0647adf6bf344803ed5);
        assert_eq!(seq.parameters[999994], 0x80001ffae238e372bf5e3efe7339c557);
        assert_eq!(seq.parameters[999998], 0x800008b8c8845531179837fa8ad9c164);
        assert_eq!(seq.parameters[999999], 0x800002e842c03d050808f7952847fac9);
    }

    #[test]
    fn check_sequence_samples_5d() {
        let mut seq = Sequence::new(5);

        assert_eq!(seq.next_raw(), 0x0);
        assert_eq!(seq.next_raw(), 0x0);
        assert_eq!(seq.next_raw(), 0x0);
        assert_eq!(seq.next_raw(), 0x0);
        assert_eq!(seq.next_raw(), 0x0);

        seq.seek(1_000);

        assert_eq!(seq.next_raw(), 0x457e82764cd63809);
        assert_eq!(seq.next_raw(), 0xa3af00ddf92ab278);
        assert_eq!(seq.next_raw(), 0x6e1cf8f61691a5b1);
        assert_eq!(seq.next_raw(), 0x2b3296eeb51a048c);
        assert_eq!(seq.next_raw(), 0x8e2e8b7198b4d3a5);

        seq.seek(1_000_000);

        assert_eq!(seq.next_raw(), 0x762d9e1c24cae389);
        assert_eq!(seq.next_raw(), 0x639b63154ec92866);
        assert_eq!(seq.next_raw(), 0x212c814828ef3e1a);
        assert_eq!(seq.next_raw(), 0xbd9d94736da1c548);
        assert_eq!(seq.next_raw(), 0x65d0b3bc825abfed);

        seq.seek(1_000_000_000);

        assert_eq!(seq.next_raw(), 0xa2319defb888d0e6);
        assert_eq!(seq.next_raw(), 0x16fb0b3bc1c5cf6e);
        assert_eq!(seq.next_raw(), 0x95d901dfe68a9713);
        assert_eq!(seq.next_raw(), 0xaf8be2e43feaa1b7);
        assert_eq!(seq.next_raw(), 0xb73e185d327db5e3);

        seq.seek(1_000_000_000_000);

        assert_eq!(seq.next_raw(), 0x91d0f068d67002b3);
        assert_eq!(seq.next_raw(), 0xc4a3e16cecb2461c);
        assert_eq!(seq.next_raw(), 0x57af529c8d5e25a8);
        assert_eq!(seq.next_raw(), 0xba6e4b99ac87b38c);
        assert_eq!(seq.next_raw(), 0xca8f2c0d3b0e7f45);

        seq.seek(1_000_000_000_000_000);

        assert_eq!(seq.next_raw(), 0x982b1985a58a8ebf);
        assert_eq!(seq.next_raw(), 0x2028917c9861dfd1);
        assert_eq!(seq.next_raw(), 0x84dab38837c31b23);
        assert_eq!(seq.next_raw(), 0x3ed75049f2155e95);
        assert_eq!(seq.next_raw(), 0x3f4413aeb0a12689);

        seq.seek(1_000_000_000_000_000_000);

        assert_eq!(seq.next_raw(), 0x685bb20ea53d9b28);
        assert_eq!(seq.next_raw(), 0x9e784eb33e524b91);
        assert_eq!(seq.next_raw(), 0xf64d4c19d2220204);
        assert_eq!(seq.next_raw(), 0x7911a0d9a37979a6);
        assert_eq!(seq.next_raw(), 0x21ece261f57e87da);
    }
}
