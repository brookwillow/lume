"""Download and validate the local speech-recognition models."""

from .voice import preload_model


def main() -> None:
    preload_model()


if __name__ == "__main__":
    main()
