module.exports = {
  title: `Sun-chive`,
  description: `우물 밖을 탈출하기 위한 기록`,
  language: `ko`, // `ko`, `en` => currently support versions for Korean and English
  siteUrl: `https://sunkyoung.github.io/`,
  ogImage: `/og-image.png`, // Path to your in the 'static' folder
  comments: {
    utterances: {
      repo: ``, // `zoomkoding/zoomkoding-gatsby-blog`,
    },
  },
  ga: '0', // Google Analytics Tracking ID
  author: {
    name: `김선경`,
    bio: {
      role: `NLP Researcher`,
      description: ['데이터 기반의 사고를 가진', '더 나은 세상을 만들고 싶은', '끊임없이 도전하는'],
      thumbnail: 'sample.png', // Path to the image in the 'asset' folder
    },
    social: {
      github: `https://github.com/Sunkyoung`, // `https://github.com/zoomKoding`,
      linkedIn: `https://www.linkedin.com/in/sunkyoung-kim`, // `https://www.linkedin.com/in/jinhyeok-jeong-800871192`,
      email: `sunkyoung@kaist.ac.kr`, // `zoomkoding@gmail.com`,
      CV: ``,
    },
  },

  // metadata for About Page
  about: {
    timestamps: [
      // =====       [Timestamp Sample and Structure]      =====
      // ===== 🚫 Don't erase this sample (여기 지우지 마세요!) =====
      {
        date: '',
        activity: '',
        links: {
          github: '',
          post: '',
          googlePlay: '',
          appStore: '',
          demo: '',
        },
      },
      // ========================================================
      // ========================================================
    ],

    projects: [
      // =====        [Project Sample and Structure]        =====
      // ===== 🚫 Don't erase this sample (여기 지우지 마세요!)  =====
      {
        title: '',
        description: '',
        techStack: ['', ''],
        thumbnailUrl: '',
        links: {
          post: '',
          github: '',
          googlePlay: '',
          appStore: '',
          demo: '',
        },
      },
      // ========================================================
      // ========================================================
    ],
  },
};
