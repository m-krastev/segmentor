from .main import main


if __name__ == "__main__":
    # from cProfile import Profile
    # from pstats import Stats

    # # Profile the main function
    # with Profile() as pr:
    #     try:
    #         main()
    #     except:
    #         pass
    #     (Stats(pr).strip_dirs().sort_stats("cumulative").print_stats(10))
    #     (Stats(pr).dump_stats("profile_output.prof"))

    main()