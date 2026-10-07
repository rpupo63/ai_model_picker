# Maintainer: Beto
pkgname=ai-model-picker
pkgver=0.1.0
pkgrel=1
pkgdesc="Unified AI model provider selection and configuration library"
arch=('any')
url="https://github.com/beto/ai-model-picker"
license=('MIT')
depends=('python' 'python-inquirerpy')
makedepends=('python-build' 'python-installer' 'python-wheel' 'python-setuptools')
source=()
sha256sums=()

build() {
    cd "$startdir"
    python -m build --wheel --no-isolation
}

package() {
    cd "$startdir"
    python -m installer --destdir="$pkgdir" dist/*.whl

    install -Dm644 LICENSE "$pkgdir/usr/share/licenses/$pkgname/LICENSE"
}
