from manuscript.tables import prepare_agreement as agreement
from manuscript.tables import prepare_performance as performance


def main() -> None:
    performance.prepare()
    agreement.prepare()


if __name__ == "__main__":
    main()
