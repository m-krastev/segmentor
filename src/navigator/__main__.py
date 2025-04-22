from .main import main


if __name__ == "__main__":
    from cProfile import Profile
    from pstats import Stats

    # Profile the main function
    with Profile() as pr:
        try:
            main()
        except Exception as e:
            raise e
        finally:
            (Stats(pr).sort_stats("cumulative").print_stats(100))
            (Stats(pr).dump_stats("profile_output.prof"))

    # main()